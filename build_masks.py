import cv2
import numpy as np
import os
import argparse

# Les calques sont calculés une seule fois, sur l'image médiane de toutes les dates :
# les images alignées étant superposables au pixel près, ils valent pour toute la série.

def load_aligned_images(input_dir):
    """
    Charge toutes les images alignées d'un dossier (triées par nom, donc par date).
    """
    files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    images = [cv2.imread(os.path.join(input_dir, f)) for f in files]
    return files, [im for im in images if im is not None]

def median_image(images, rows_per_block=200):
    """
    Image médiane pixel par pixel, calculée par bandes pour limiter la mémoire.
    """
    height, width = images[0].shape[:2]
    median = np.zeros_like(images[0])
    for y in range(0, height, rows_per_block):
        block = np.stack([im[y:y + rows_per_block] for im in images])
        median[y:y + rows_per_block] = np.median(block, axis=0).astype(np.uint8)
    return median

def retina_mask(images):
    """
    Zone visible sur toutes les dates (le cercle commun produit par align_images.py).
    """
    mask = np.ones(images[0].shape[:2], dtype=bool)
    for im in images:
        mask &= im.max(axis=-1) > 0
    return mask.astype(np.uint8) * 255

def border_mask(retina, width):
    """
    Couronne sur le bord du disque rétinien, où les effets de bord sont fréquents.
    """
    inner = cv2.erode(retina, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * width + 1, 2 * width + 1)))
    return cv2.subtract(retina, inner)

def dark_structures(channel):
    """
    Réponse black-hat multi-échelle : structures plus sombres que leur voisinage.
    """
    channel = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(channel)
    response = np.zeros(channel.shape, dtype=np.float32)
    for size in (11, 21, 31):
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
        response = np.maximum(response, cv2.morphologyEx(channel, cv2.MORPH_BLACKHAT, kernel).astype(np.float32))
    return cv2.GaussianBlur(response, (0, 0), 1.0)

def vessel_mask(reference, retina, margin, red_weight=0.8):
    """
    Vaisseaux : les vaisseaux sont rouges, donc sombres dans le canal vert mais presque
    invisibles dans le canal rouge. Les espaces entre druses voisines, eux, sont sombres
    dans les deux canaux : on soustrait donc la réponse du rouge à celle du vert, ce qui
    évite de prendre ces espaces pour des vaisseaux.
    Seules les composantes assez étendues sont gardées.
    """
    inside = cv2.erode(retina, np.ones((15, 15), np.uint8)) > 0
    green = dark_structures(reference[..., 1])
    red = dark_structures(reference[..., 2])
    green /= np.percentile(green[inside], 99)
    red /= np.percentile(red[inside], 99)
    response = np.clip(green - red_weight * red, 0, None)

    threshold = np.percentile(response[inside], 88)
    vessels = ((response > threshold) & inside).astype(np.uint8)

    count, labels, stats, _ = cv2.connectedComponentsWithStats(vessels, connectivity=8)
    keep = np.zeros(count, dtype=bool)
    for i in range(1, count):
        extent = max(stats[i, cv2.CC_STAT_WIDTH], stats[i, cv2.CC_STAT_HEIGHT])
        keep[i] = stats[i, cv2.CC_STAT_AREA] >= 150 and extent > 120
    vessels = keep[labels].astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * margin + 1, 2 * margin + 1))
    return cv2.dilate(vessels, kernel)

def disk_kernel(radius):
    kernel = np.zeros((2 * radius + 1, 2 * radius + 1), dtype=np.float32)
    cv2.circle(kernel, (radius, radius), radius, 1.0, -1)
    return kernel

def find_optic_disc(reference, retina, radii=range(60, 181, 10), scale=4):
    """
    Papille : disque clair. Pour chaque rayon candidat, on compare en tout point la
    luminosité moyenne à l'intérieur d'un disque à celle de l'anneau qui l'entoure ;
    la papille est le disque le plus contrasté. Les moyennes ne tiennent compte que
    des pixels de la rétine, ce qui permet de trouver une papille proche du bord.
    On travaille sur le canal rouge, où la papille est uniformément claire (dans le vert,
    seule son excavation centrale ressort), après avoir effacé les vaisseaux qui la
    traversent par une fermeture morphologique.
    Retourne son centre (x, y) et son rayon.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (31, 31))
    lightness = cv2.morphologyEx(reference[..., 2], cv2.MORPH_CLOSE, kernel)
    small_size = (lightness.shape[1] // scale, lightness.shape[0] // scale)
    small = cv2.resize(lightness, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
    mask = (cv2.resize(retina, small_size, interpolation=cv2.INTER_NEAREST) > 0).astype(np.float32)
    small *= mask

    best = (-np.inf, None, None)
    for radius in radii:
        r = max(radius // scale, 2)
        inner, outer = disk_kernel(r), disk_kernel(int(r * 1.6))
        ring = outer.copy()
        ring[outer.shape[0] // 2 - r:outer.shape[0] // 2 + r + 1, outer.shape[1] // 2 - r:outer.shape[1] // 2 + r + 1] -= inner
        sum_in, count_in = cv2.filter2D(small, -1, inner), cv2.filter2D(mask, -1, inner)
        sum_ring, count_ring = cv2.filter2D(small, -1, ring), cv2.filter2D(mask, -1, ring)
        score = sum_in / np.maximum(count_in, 1) - sum_ring / np.maximum(count_ring, 1)
        # Le disque doit être entièrement dans la rétine, l'anneau au moins à moitié
        score[(count_in < inner.sum() * 0.98) | (count_ring < ring.sum() * 0.5)] = -np.inf
        _, value, _, location = cv2.minMaxLoc(score)
        if value > best[0]:
            best = (value, location, radius)

    _, location, radius = best
    return (location[0] * scale + scale // 2, location[1] * scale + scale // 2), radius

def find_fovea(reference, retina, disc_center, disc_radius):
    """
    Fovéa : point le plus sombre de la macula, cherché à environ 2,5 diamètres
    papillaires de la papille, du côté du centre de l'image.
    """
    lightness = cv2.cvtColor(reference, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    blurred = cv2.GaussianBlur(lightness, (0, 0), 30)
    height, width = lightness.shape
    yy, xx = np.mgrid[0:height, 0:width]
    disc_diameter = 2 * disc_radius
    distance = np.hypot(xx - disc_center[0], yy - disc_center[1])
    direction = 1 if disc_center[0] > width / 2 else -1  # la fovéa est du côté opposé au bord le plus proche
    toward_center = (disc_center[0] - xx) * direction > 0
    search = (distance > 1.8 * disc_diameter) & (distance < 3.2 * disc_diameter) & toward_center
    search &= np.abs(yy - disc_center[1]) < 1.0 * disc_diameter
    search &= cv2.erode(retina, np.ones((61, 61), np.uint8)) > 0
    candidates = np.where(search, blurred, np.inf)
    y, x = np.unravel_index(np.argmin(candidates), candidates.shape)
    return (int(x), int(y))

def save_control_image(reference, layers, output_path):
    """
    Image de contrôle : la référence avec chaque calque en couleur semi-transparente.
    """
    overlay = reference.copy()
    colors = {'vessels': (255, 80, 0), 'optic_disc': (0, 255, 0), 'fovea': (255, 255, 0), 'border': (0, 200, 255)}
    for name, color in colors.items():
        overlay[layers[name] > 0] = color
    control = cv2.addWeighted(reference, 0.55, overlay, 0.45, 0)
    for i, (name, color) in enumerate(colors.items()):
        cv2.putText(control, name, (30, 60 + 50 * i), cv2.FONT_HERSHEY_SIMPLEX, 1.4, color, 3)
    cv2.imwrite(output_path, control, [cv2.IMWRITE_JPEG_QUALITY, 92])

def build_masks(input_dir, output_dir, vessel_margin=4, disc_margin=1.3, fovea_ratio=0.35, border_width=30):
    os.makedirs(output_dir, exist_ok=True)
    files, images = load_aligned_images(input_dir)
    if not images:
        print(f"Aucune image trouvée dans le dossier : {input_dir}")
        return
    print(f"{len(images)} images chargées.")

    print("Calcul de l'image de référence (médiane de toutes les dates)...")
    reference = median_image(images)
    retina = retina_mask(images)
    del images

    print("Calcul des calques...")
    disc_center, disc_radius = find_optic_disc(reference, retina)
    fovea_center = find_fovea(reference, retina, disc_center, disc_radius)
    print(f"  -> Papille : centre {disc_center}, rayon {disc_radius} px")
    print(f"  -> Fovéa : centre {fovea_center}")

    layers = {
        'vessels': vessel_mask(reference, retina, vessel_margin),
        'optic_disc': np.zeros_like(retina),
        'fovea': np.zeros_like(retina),
        'border': border_mask(retina, border_width),
    }
    cv2.circle(layers['optic_disc'], disc_center, int(disc_radius * disc_margin), 255, -1)
    cv2.circle(layers['fovea'], fovea_center, int(2 * disc_radius * fovea_ratio), 255, -1)

    # Zone d'analyse : la rétine, moins tous les calques d'exclusion
    excluded = np.zeros_like(retina)
    for layer in layers.values():
        excluded = cv2.bitwise_or(excluded, layer)
    analysis = cv2.bitwise_and(retina, cv2.bitwise_not(excluded))
    print(f"  -> Zone analysée : {100 * np.count_nonzero(analysis) / np.count_nonzero(retina):.0f} % de la rétine")

    cv2.imwrite(os.path.join(output_dir, 'reference.png'), reference)
    cv2.imwrite(os.path.join(output_dir, 'retina.png'), retina)
    for name, layer in layers.items():
        cv2.imwrite(os.path.join(output_dir, f'{name}.png'), layer)
    cv2.imwrite(os.path.join(output_dir, 'analysis_zone.png'), analysis)
    save_control_image(reference, layers, os.path.join(output_dir, 'control.jpg'))
    print(f"Calques sauvegardés dans : {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Calcule les calques d'exclusion (vaisseaux, papille, fovéa, bord) "
                                                 "à partir des images alignées.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les calques.")
    parser.add_argument('--vessel_margin', type=int, default=4, help="Marge (px) ajoutée autour des vaisseaux.")
    parser.add_argument('--disc_margin', type=float, default=1.3, help="Facteur d'agrandissement du masque de la papille.")
    parser.add_argument('--fovea_ratio', type=float, default=0.35,
                        help="Rayon du masque fovéal, en fraction du diamètre papillaire.")
    parser.add_argument('--border_width', type=int, default=30, help="Largeur (px) de la couronne exclue au bord.")
    args = parser.parse_args()
    build_masks(args.input_dir, args.output_dir, args.vessel_margin, args.disc_margin, args.fovea_ratio, args.border_width)
