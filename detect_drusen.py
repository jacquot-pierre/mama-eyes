import cv2
import numpy as np
import os
import argparse
import csv
import re

# Le score de druse est enregistré en PNG 8 bits : valeur = contraste relatif × SCORE_SCALE
SCORE_SCALE = 1000
# Idem pour la densité de druses fines (valeurs typiques 0,01 à 0,06)
DENSITY_SCALE = 4000

def local_background(channel, retina, size=101, scale=4):
    """
    Niveau typique du voisinage de chaque pixel : médiane locale sur une fenêtre
    d'environ size pixels (calculée à résolution réduite). Une médiane, contrairement
    à un minimum ou une moyenne, n'est tirée ni par les druses ni par les zones sombres.
    Hors rétine, on remplit avec la médiane globale pour ne pas tirer le fond vers le noir.
    """
    height, width = channel.shape
    filled = np.where(retina, channel, int(np.median(channel[retina]))).astype(np.uint8)
    small = cv2.resize(filled, (width // scale, height // scale), interpolation=cv2.INTER_AREA)
    median = cv2.medianBlur(small, (size // scale) | 1)
    return cv2.resize(median, (width, height), interpolation=cv2.INTER_LINEAR).astype(np.float32)

def masked_blur(image, mask, sigma):
    """
    Flou gaussien normalisé par le masque : seuls les pixels du masque contribuent.
    """
    m = mask.astype(np.float32)
    num = cv2.GaussianBlur(image * m, (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)

def fundus_background(smooth, green, vessels, retina, scale=4, sigma=12, iterations=2):
    """
    Niveau de la rétine « ordinaire » (ni druse, ni tache sombre) autour de chaque pixel.
    Dans les zones presque couvertes de druses, une médiane locale serait au niveau des
    druses elles-mêmes, qui ne ressortiraient alors qu'à peine. On part donc de la médiane
    locale, puis on recalcule le fond uniquement à partir des pixels proches de ce niveau
    (en excluant les druses et les zones sombres comme l'atrophie), à résolution réduite.
    """
    height, width = smooth.shape
    small_size = (width // scale, height // scale)
    small = cv2.resize(smooth, small_size, interpolation=cv2.INTER_AREA)
    small_retina = cv2.resize(retina.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST) > 0
    small_vessels = cv2.resize(vessels.astype(np.uint8), small_size, interpolation=cv2.INTER_NEAREST) > 0
    background = cv2.resize(local_background(green, retina), small_size, interpolation=cv2.INTER_AREA)
    for _ in range(iterations):
        contrast = (small - background) / np.maximum(background, 10)
        ordinary = small_retina & ~small_vessels & (contrast < 0.03) & (contrast > -0.10)
        background = masked_blur(small, ordinary, sigma)
    return cv2.resize(background, (width, height), interpolation=cv2.INTER_LINEAR)

def green_without_vessels(image, vessels):
    """
    Canal vert (le plus contrasté pour les druses), vaisseaux effacés par fermeture
    morphologique pour ne pas faire paraître claires les zones situées entre eux.
    Retourne le canal (uint8) et sa version légèrement lissée (float32).
    """
    green = image[..., 1]
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    green = np.where(vessels, closed, green).astype(np.uint8)
    return green, cv2.GaussianBlur(green.astype(np.float32), (0, 0), 1.5)

def common_background(reference, vessels, retina):
    """
    Fond commun à toutes les dates, calculé une seule fois sur l'image médiane.
    Un fond propre à chaque date suivrait l'évolution de la rétine (atrophie qui s'étend,
    zones qui s'assombrissent) et ferait varier le score de zones inchangées. Avec un fond
    commun, et grâce à l'harmonisation des couleurs entre dates, une druse qui se résorbe
    redescend simplement au niveau du fond.
    """
    green, smooth = green_without_vessels(reference, vessels)
    return fundus_background(smooth, green, vessels, retina)

def drusen_score(image, vessels, background):
    """
    Score de druse : contraste relatif du canal vert par rapport au fond de la rétine.
    """
    _, smooth = green_without_vessels(image, vessels)
    return (smooth - background) / np.maximum(background, 10)

def fine_drusen_score(image, vessels, radius=10):
    """
    Score des petites druses : contraste relatif par rapport au voisinage immédiat
    (top-hat morphologique, disque d'environ 2 × radius px). À cette échelle, les petits
    creux entre deux druses fines suffisent à définir le fond, même dans une zone
    entièrement tapissée de druses, où le fond commun est au niveau des druses elles-mêmes.
    """
    _, smooth = green_without_vessels(image, vessels)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * radius + 1, 2 * radius + 1))
    opened = cv2.GaussianBlur(cv2.morphologyEx(smooth, cv2.MORPH_OPEN, kernel), (0, 0), radius / 2)
    return (smooth - opened) / np.maximum(opened, 10)

def temporal_median_scores(image_paths, score_function, window):
    """
    Pour chaque date, médiane pixel par pixel des scores de `window` dates consécutives
    centrées sur elle : une druse doit être vue sur la majorité de ces dates pour ressortir.
    score_function renvoie un tuple de cartes de score ; chacune est filtrée séparément.
    Les scores sont calculés au fil de l'eau pour ne garder que `window` dates en mémoire.
    """
    half = window // 2
    buffer = {}
    for i in range(len(image_paths)):
        indices = range(max(i - half, 0), min(i + half, len(image_paths) - 1) + 1)
        for j in indices:
            if j not in buffer:
                buffer[j] = tuple(s.astype(np.float32) for s in score_function(cv2.imread(image_paths[j])))
        for j in [k for k in buffer if k < i - half]:
            del buffer[j]
        neighbors = [buffer[j] for j in indices]
        yield tuple(np.median(np.stack([n[k] for n in neighbors]), axis=0) if len(neighbors) > 1 else neighbors[0][k]
                    for k in range(len(neighbors[0])))

def segment_drusen(score, zone, low, high, min_area, max_elongation):
    """
    Seuillage à deux niveaux : une druse est une région où le score dépasse `low`
    et qui contient au moins un point au-dessus de `high`. Les régions trop petites
    (bruit) ou trop allongées (liserés le long des vaisseaux) sont écartées.
    """
    candidates = ((score > low) & zone).astype(np.uint8)
    count, labels, stats, _ = cv2.connectedComponentsWithStats(candidates, connectivity=8)
    has_seed = np.zeros(count, dtype=bool)
    has_seed[np.unique(labels[(score > high) & zone])] = True

    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        if not has_seed[i] or stats[i, cv2.CC_STAT_AREA] < min_area:
            continue
        x, y = stats[i, cv2.CC_STAT_LEFT], stats[i, cv2.CC_STAT_TOP]
        w, h = stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT]
        points = np.column_stack(np.nonzero(labels[y:y + h, x:x + w] == i))
        if len(points) >= 5:
            (_, _), (a, b), _ = cv2.minAreaRect(points.astype(np.float32))
            if max(a, b) / max(min(a, b), 1) > max_elongation:
                continue
        keep[i] = True
    return keep[labels]

def date_from_filename(filename):
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    return f"{match.group(1)}_{match.group(2)}_{match.group(3)}" if match else os.path.splitext(filename)[0]

def density_heatmap(image, density, retina):
    """
    Image de contrôle de la densité : carte de chaleur fondue avec la photo
    (clair = beaucoup de druses fines, violet = peu ou pas de druses).
    """
    values = np.clip((density * DENSITY_SCALE - 60) * 1.5, 0, 255).astype(np.uint8)
    heat = cv2.applyColorMap(values, cv2.COLORMAP_INFERNO)
    blend = cv2.addWeighted(image, 0.45, heat, 0.55, 0)
    blend[~retina] = 0
    return blend

def detect_drusen(input_dir, masks_dir, output_dir, low=0.12, high=0.18, min_area=40, max_elongation=4.0,
                  temporal_window=3, fine_low=0.06, fine_high=0.10, fine_min_area=15, density_sigma=40.0):
    for sub in ('scores', 'masks', 'overlays', 'density', 'density_overlays'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    retina = cv2.imread(os.path.join(masks_dir, 'retina.png'), cv2.IMREAD_GRAYSCALE)
    vessels = cv2.imread(os.path.join(masks_dir, 'vessels.png'), cv2.IMREAD_GRAYSCALE)
    zone = cv2.imread(os.path.join(masks_dir, 'analysis_zone.png'), cv2.IMREAD_GRAYSCALE)
    if retina is None or vessels is None or zone is None or not os.path.exists(os.path.join(masks_dir, 'reference.png')):
        print(f"Calques introuvables dans {masks_dir} : lancer d'abord build_masks.py")
        return
    reference = cv2.imread(os.path.join(masks_dir, 'reference.png'))
    retina, vessels, zone = retina > 0, vessels > 0, zone > 0
    zone_area = np.count_nonzero(zone)

    print("Calcul du fond commun à partir de l'image médiane...")
    background = common_background(reference, vessels, retina)

    summary = []
    image_paths = [os.path.join(input_dir, f) for f in image_files]
    score_function = lambda image: (drusen_score(image, vessels, background), fine_drusen_score(image, vessels))
    scores = temporal_median_scores(image_paths, score_function, temporal_window)
    for filename, (score, fine_score) in zip(image_files, scores):
        image = cv2.imread(os.path.join(input_dir, filename))
        date = date_from_filename(filename)

        # Druses franches (fond commun) et petites druses (voisinage immédiat)
        large = segment_drusen(score, zone, low, high, min_area, max_elongation)
        small = segment_drusen(fine_score, zone, fine_low, fine_high, fine_min_area, max_elongation) & ~large
        drusen = large | small

        # Densité de druses fines : intensité moyenne des petites taches claires alentour
        density = masked_blur(np.clip(fine_score, 0, None), zone, density_sigma)
        density[~zone] = 0

        score_png = np.clip(np.where(zone, score, 0) * SCORE_SCALE, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'scores', f'score_{date}.png'), score_png)
        cv2.imwrite(os.path.join(output_dir, 'masks', f'drusen_{date}.png'), drusen.astype(np.uint8) * 255)
        density_png = np.clip(density * DENSITY_SCALE, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'density', f'density_{date}.png'), density_png)
        cv2.imwrite(os.path.join(output_dir, 'density_overlays', f'density_{date}.jpg'),
                    density_heatmap(image, density, retina), [cv2.IMWRITE_JPEG_QUALITY, 90])

        overlay = image.copy()
        for mask, color in ((small, (0, 255, 255)), (large, (0, 255, 0))):
            contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(overlay, contours, -1, color, 2)
        cv2.imwrite(os.path.join(output_dir, 'overlays', f'drusen_{date}.jpg'), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        area = int(np.count_nonzero(drusen))
        count = cv2.connectedComponents(drusen.astype(np.uint8))[0] - 1
        mean_density = float(density[zone].mean())
        summary.append({'date': date, 'nb_druses': count, 'surface_px': area,
                        'surface_pct_zone': round(100 * area / zone_area, 2),
                        'surface_petites_pct_zone': round(100 * np.count_nonzero(small) / zone_area, 2),
                        'densite_moyenne': round(mean_density, 5)})
        print(f"{date} : {count} druses, {100 * area / zone_area:.1f} % de la zone analysée, densité {mean_density:.4f}")

    with open(os.path.join(output_dir, 'drusen_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'nb_druses', 'surface_px', 'surface_pct_zone',
                                               'surface_petites_pct_zone', 'densite_moyenne'])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Résultats sauvegardés dans : {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Détecte les druses sur chaque image alignée.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--masks_dir', type=str, required=True, help="Dossier des calques produits par build_masks.py.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les résultats.")
    parser.add_argument('--low', type=float, default=0.12, help="Seuil bas de contraste relatif (contour des druses).")
    parser.add_argument('--high', type=float, default=0.18, help="Seuil haut : chaque druse doit contenir un point au-dessus.")
    parser.add_argument('--temporal_window', type=int, default=3,
                        help="Nombre de dates consécutives dont on prend la médiane du score (1 = aucune).")
    parser.add_argument('--min_area', type=int, default=40, help="Surface minimale d'une druse (px).")
    parser.add_argument('--max_elongation', type=float, default=4.0,
                        help="Allongement maximal (longueur / largeur) : au-delà, la détection est écartée.")
    parser.add_argument('--fine_low', type=float, default=0.06, help="Seuil bas du détecteur de petites druses.")
    parser.add_argument('--fine_high', type=float, default=0.10, help="Seuil haut du détecteur de petites druses.")
    parser.add_argument('--fine_min_area', type=int, default=15, help="Surface minimale d'une petite druse (px).")
    parser.add_argument('--density_sigma', type=float, default=40.0,
                        help="Taille (px) de la fenêtre sur laquelle est mesurée la densité de druses fines.")
    args = parser.parse_args()
    detect_drusen(args.input_dir, args.masks_dir, args.output_dir, args.low, args.high, args.min_area,
                  args.max_elongation, args.temporal_window, args.fine_low, args.fine_high, args.fine_min_area,
                  args.density_sigma)
