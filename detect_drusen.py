import cv2
import numpy as np
import os
import argparse
import csv
import re

# Le score de druse est enregistré en PNG 8 bits : valeur = contraste relatif × SCORE_SCALE
SCORE_SCALE = 1000

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

def drusen_score(image, vessels, retina):
    """
    Score de druse : contraste relatif du canal vert par rapport au fond local.
    Les druses sont des taches claires ; les vaisseaux (sombres) sont d'abord effacés
    pour ne pas faire paraître claires les zones situées entre eux.
    """
    green = image[..., 1]
    closed = cv2.morphologyEx(green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (21, 21)))
    green = np.where(vessels, closed, green).astype(np.uint8)
    background = local_background(green, retina)
    smooth = cv2.GaussianBlur(green.astype(np.float32), (0, 0), 1.5)
    return (smooth - background) / np.maximum(background, 10)

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

def detect_drusen(input_dir, masks_dir, output_dir, low=0.07, high=0.12, min_area=40, max_elongation=4.0):
    for sub in ('scores', 'masks', 'overlays'):
        os.makedirs(os.path.join(output_dir, sub), exist_ok=True)
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    retina = cv2.imread(os.path.join(masks_dir, 'retina.png'), cv2.IMREAD_GRAYSCALE)
    vessels = cv2.imread(os.path.join(masks_dir, 'vessels.png'), cv2.IMREAD_GRAYSCALE)
    zone = cv2.imread(os.path.join(masks_dir, 'analysis_zone.png'), cv2.IMREAD_GRAYSCALE)
    if retina is None or vessels is None or zone is None:
        print(f"Calques introuvables dans {masks_dir} : lancer d'abord build_masks.py")
        return
    retina, vessels, zone = retina > 0, vessels > 0, zone > 0
    zone_area = np.count_nonzero(zone)

    summary = []
    for filename in image_files:
        image = cv2.imread(os.path.join(input_dir, filename))
        if image is None: continue
        date = date_from_filename(filename)

        score = drusen_score(image, vessels, retina)
        drusen = segment_drusen(score, zone, low, high, min_area, max_elongation)

        score_png = np.clip(np.where(zone, score, 0) * SCORE_SCALE, 0, 255).astype(np.uint8)
        cv2.imwrite(os.path.join(output_dir, 'scores', f'score_{date}.png'), score_png)
        cv2.imwrite(os.path.join(output_dir, 'masks', f'drusen_{date}.png'), drusen.astype(np.uint8) * 255)

        overlay = image.copy()
        contours, _ = cv2.findContours(drusen.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(overlay, contours, -1, (0, 255, 0), 2)
        cv2.imwrite(os.path.join(output_dir, 'overlays', f'drusen_{date}.jpg'), overlay, [cv2.IMWRITE_JPEG_QUALITY, 90])

        area = int(np.count_nonzero(drusen))
        summary.append({'date': date, 'nb_druses': len(contours), 'surface_px': area,
                        'surface_pct_zone': round(100 * area / zone_area, 2)})
        print(f"{date} : {len(contours)} druses, {100 * area / zone_area:.1f} % de la zone analysée")

    with open(os.path.join(output_dir, 'drusen_summary.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['date', 'nb_druses', 'surface_px', 'surface_pct_zone'])
        writer.writeheader()
        writer.writerows(summary)
    print(f"Résultats sauvegardés dans : {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Détecte les druses sur chaque image alignée.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--masks_dir', type=str, required=True, help="Dossier des calques produits par build_masks.py.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les résultats.")
    parser.add_argument('--low', type=float, default=0.07, help="Seuil bas de contraste relatif (contour des druses).")
    parser.add_argument('--high', type=float, default=0.12, help="Seuil haut : chaque druse doit contenir un point au-dessus.")
    parser.add_argument('--min_area', type=int, default=40, help="Surface minimale d'une druse (px).")
    parser.add_argument('--max_elongation', type=float, default=4.0,
                        help="Allongement maximal (longueur / largeur) : au-delà, la détection est écartée.")
    args = parser.parse_args()
    detect_drusen(args.input_dir, args.masks_dir, args.output_dir, args.low, args.high, args.min_area, args.max_elongation)
