import cv2
import numpy as np
import os
import argparse

# Facteur de réduction utilisé pour estimer les champs d'éclairage (basse fréquence)
ILLUM_DOWNSCALE = 4

def find_retina_circle(image):
    """
    Trouve le cercle principal de la rétine dans une image avec une méthode plus robuste.
    Retourne le centre (x, y) et le rayon du cercle.
    """
    image_blur = cv2.GaussianBlur(image, (0,0), 3)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None, None
    largest_contour = max(contours, key=cv2.contourArea)
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)
    radius = radius * 0.90
    return (int(x), int(y)), int(radius)

def color_transfer_to_target(source_image, source_mask, target_stats):
    """
    Normalise la couleur de l'image source vers des statistiques cibles (moyenne globale).
    """
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    source_l, source_a, source_b = cv2.split(source_lab)

    source_l_mean, source_l_std = cv2.meanStdDev(source_l, mask=source_mask)
    source_a_mean, source_a_std = cv2.meanStdDev(source_a, mask=source_mask)
    source_b_mean, source_b_std = cv2.meanStdDev(source_b, mask=source_mask)

    # Normaliser chaque canal vers la cible globale
    l_normalized = (source_l - source_l_mean) * (target_stats['l_std'] / source_l_std) + target_stats['l_mean']
    a_normalized = (source_a - source_a_mean) * (target_stats['a_std'] / source_a_std) + target_stats['a_mean']
    b_normalized = (source_b - source_b_mean) * (target_stats['b_std'] / source_b_std) + target_stats['b_mean']

    l_normalized = np.clip(l_normalized, 0, 255)
    a_normalized = np.clip(a_normalized, 0, 255)
    b_normalized = np.clip(b_normalized, 0, 255)

    lab_normalized = cv2.merge([l_normalized.astype('uint8'), a_normalized.astype('uint8'), b_normalized.astype('uint8')])
    bgr_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)
    return bgr_normalized

def apply_clahe(image, mask):
    """
    Applique un CLAHE sur la luminance (canal L) puis remasque l'image.
    """
    im_masked = cv2.bitwise_and(image, image, mask=mask)
    lab = cv2.cvtColor(im_masked, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe = clahe_final.apply(l)
    lab_clahe = cv2.merge((l_clahe, a, b))
    final_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
    return cv2.bitwise_and(final_image, final_image, mask=mask)

def common_circle_mask(valid_masks):
    """
    D'une visite à l'autre, l'œil est décalé dans le cercle de l'appareil photo : une fois
    la rétine recalée, les disques photographiés ne se superposent plus. Découper toutes
    les images avec le disque d'une seule photo produirait alors des formes en lentille
    (impression d'ellipse) différentes à chaque date.
    On utilise donc le plus grand cercle contenu dans la zone visible sur toutes les dates.
    """
    common = valid_masks[0].copy()
    for mask in valid_masks[1:]:
        common = cv2.bitwise_and(common, mask)
    distance = cv2.distanceTransform((common > 0).astype(np.uint8), cv2.DIST_L2, 5)
    _, radius, _, center = cv2.minMaxLoc(distance)
    print(f"Cercle commun à toutes les dates : centre {center}, rayon {int(radius)} px")
    circle_mask = np.zeros_like(common)
    cv2.circle(circle_mask, center, int(radius), 255, -1)
    return circle_mask

def masked_blur(image, mask, sigma):
    """
    Flou gaussien normalisé par le masque : les pixels hors masque (fond noir,
    coins vides après recalage) ne tirent pas la moyenne vers le noir.
    """
    m = (mask > 0).astype(np.float32)
    num = cv2.GaussianBlur(image * m[..., None], (0, 0), sigma)
    den = cv2.GaussianBlur(m, (0, 0), sigma)
    return num / np.maximum(den, 1e-3)[..., None]

def match_histograms_lab(image, mask, target_cdfs):
    """
    Fait correspondre l'histogramme de chaque canal Lab (dans le masque) aux CDF cibles.
    """
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    channels = list(cv2.split(lab))
    for c in range(3):
        hist = cv2.calcHist([channels[c]], [0], mask, [256], [0, 256]).ravel()
        cdf = np.cumsum(hist) / max(hist.sum(), 1)
        lut = np.interp(cdf, target_cdfs[c], np.arange(256))
        channels[c] = cv2.LUT(channels[c], np.clip(lut, 0, 255).astype(np.uint8))
    return cv2.cvtColor(cv2.merge(channels), cv2.COLOR_LAB2BGR)

def median_ignoring_nan(stack):
    """
    Médiane le long de l'axe 0 en ignorant les NaN, sans la surconsommation
    mémoire de np.nanmedian (les NaN sont rangés en fin de tri par np.sort).
    """
    sorted_stack = np.sort(stack, axis=0)
    counts = np.sum(~np.isnan(stack), axis=0)
    low = np.take_along_axis(sorted_stack, np.maximum((counts - 1) // 2, 0)[None], axis=0)[0]
    high = np.take_along_axis(sorted_stack, np.maximum(counts // 2, 0)[None], axis=0)[0]
    median = (low + high) / 2
    median[counts == 0] = np.nan
    return median

def polynomial_basis(xs, ys, degree):
    """Monômes x^i * y^j (i + j <= degree) évalués aux coordonnées données."""
    return np.stack([xs ** i * ys ** j for i in range(degree + 1) for j in range(degree + 1 - i)], axis=-1)

def fit_smooth_field(values, mask, degree, iterations=6, outlier_k=2.5):
    """
    Ajuste de façon robuste une surface polynomiale lisse sur une carte 2D
    (un canal du log-ratio image / référence). À chaque itération, les pixels dont
    l'écart à la surface dépasse outlier_k écarts-types robustes (MAD) sont exclus :
    les lésions qui évoluent réellement n'influencent donc pas la correction.
    Retourne la surface évaluée sur toute la carte.
    """
    h, w = values.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    xx = (xx - w / 2) / (w / 2)
    yy = (yy - h / 2) / (h / 2)

    # Sous-échantillonnage des points de la rétine pour accélérer l'ajustement
    points = np.zeros_like(mask, dtype=bool)
    points[::3, ::3] = True
    points &= (mask > 0) & np.isfinite(values)
    basis = polynomial_basis(xx[points], yy[points], degree)
    target = values[points]

    keep = np.ones(len(target), dtype=bool)
    for _ in range(iterations):
        coeffs, *_ = np.linalg.lstsq(basis[keep], target[keep], rcond=None)
        residuals = target - basis @ coeffs
        mad = np.median(np.abs(residuals[keep] - np.median(residuals[keep]))) * 1.4826
        keep = np.abs(residuals) < outlier_k * max(mad, 1e-4)

    return polynomial_basis(xx, yy, degree) @ coeffs

def normalize_to_reference(aligned_images, valid_masks, illum_degree, hist_match):
    """
    Uniformise l'éclairage de toutes les images en s'appuyant sur le fait qu'elles
    sont superposables pixel à pixel :
      1. image de référence = médiane pixel par pixel de toutes les dates (version réduite) ;
      2. pour chaque image et chaque canal, le rapport image / référence est modélisé
         par une surface polynomiale lisse, ajustée en ignorant les zones qui diffèrent
         réellement (lésions, atrophie) ;
      3. l'image est divisée par cette surface : les ombres, le vignettage et les
         dominantes de couleur propres à la prise de vue disparaissent, tandis que
         les changements localisés de la rétine sont conservés ;
      4. (optionnel) histogram matching Lab vers l'histogramme cumulé de toutes les images.
    Les images sont modifiées en place.
    """
    height, width = aligned_images[0].shape[:2]
    small_size = (width // ILLUM_DOWNSCALE, height // ILLUM_DOWNSCALE)

    print("  -> Réduction des images...")
    smalls = np.full((len(aligned_images), small_size[1], small_size[0], 3), np.nan, dtype=np.float32)
    small_masks = []
    for i, (image, mask) in enumerate(zip(aligned_images, valid_masks)):
        small = cv2.resize(image, small_size, interpolation=cv2.INTER_AREA).astype(np.float32)
        small_mask = cv2.resize(mask, small_size, interpolation=cv2.INTER_NEAREST)
        small = masked_blur(small, small_mask, 2.0)  # léger débruitage
        smalls[i][small_mask > 0] = small[small_mask > 0]
        small_masks.append(small_mask)

    print("  -> Calcul de l'image de référence (médiane temporelle)...")
    reference = median_ignoring_nan(smalls)

    print("  -> Correction de l'éclairage de chaque image...")
    log_reference = np.log(np.maximum(reference, 1.0))
    for i, (image, small_mask) in enumerate(zip(aligned_images, small_masks)):
        log_ratio = np.log(np.maximum(smalls[i], 1.0)) - log_reference
        fit_mask = ((small_mask > 0) & np.isfinite(log_ratio).all(axis=-1)).astype(np.uint8)
        field = np.stack([fit_smooth_field(log_ratio[..., c], fit_mask, illum_degree) for c in range(3)], axis=-1)
        gain = np.clip(np.exp(-field), 0.25, 4.0).astype(np.float32)
        gain = cv2.resize(gain, (width, height), interpolation=cv2.INTER_LINEAR)
        aligned_images[i] = np.clip(image.astype(np.float32) * gain, 0, 255).astype(np.uint8)

    if hist_match:
        print("  -> Histogram matching vers l'histogramme commun...")
        total_hists = np.zeros((3, 256), dtype=np.float64)
        for image, mask in zip(aligned_images, valid_masks):
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            for c in range(3):
                hist = cv2.calcHist([lab], [c], mask, [256], [0, 256]).ravel()
                total_hists[c] += hist / max(hist.sum(), 1)
        target_cdfs = [np.cumsum(h) / h.sum() for h in total_hists]
        for i, (image, mask) in enumerate(zip(aligned_images, valid_masks)):
            aligned_images[i] = match_histograms_lab(image, mask, target_cdfs)

def align_images(input_dir, output_dir, color_mode='reference', illum_degree=3, hist_match=False, clahe=None):
    """
    Aligne et normalise les images.
    color_mode = 'global'    : transfert de couleur vers la moyenne globale + CLAHE (ancienne méthode)
    color_mode = 'reference' : correction locale de l'éclairage vers une référence médiane
    """
    if clahe is None:
        clahe = color_mode == 'global'

    os.makedirs(output_dir, exist_ok=True)
    try:
        all_image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    # --- PASSE 1: Calculer les statistiques de couleur globales ---
    print("--- Passe 1/2: Calcul des statistiques de couleur globales...")
    all_stats = {'l_means': [], 'l_stds': [], 'a_means': [], 'a_stds': [], 'b_means': [], 'b_stds': []}
    for filename in all_image_files:
        image = cv2.imread(os.path.join(input_dir, filename))
        if image is None: continue

        center, radius = find_retina_circle(image)
        if center is None: continue

        mask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.circle(mask, center, radius, 255, -1)

        lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)

        l_mean, l_std = cv2.meanStdDev(l, mask=mask)
        a_mean, a_std = cv2.meanStdDev(a, mask=mask)
        b_mean, b_std = cv2.meanStdDev(b, mask=mask)

        all_stats['l_means'].append(l_mean[0][0])
        all_stats['l_stds'].append(l_std[0][0])
        all_stats['a_means'].append(a_mean[0][0])
        all_stats['a_stds'].append(a_std[0][0])
        all_stats['b_means'].append(b_mean[0][0])
        all_stats['b_stds'].append(b_std[0][0])

    target_stats = {
        'l_mean': np.mean(all_stats['l_means']),
        'l_std': np.mean(all_stats['l_stds']),
        'a_mean': np.mean(all_stats['a_means']),
        'a_std': np.mean(all_stats['a_stds']),
        'b_mean': np.mean(all_stats['b_means']),
        'b_std': np.mean(all_stats['b_stds'])
    }
    print("Statistiques globales calculées.")

    # --- Préparation pour l'alignement géométrique ---
    reference_filename_geom = all_image_files[0]
    im_ref_geom = cv2.imread(os.path.join(input_dir, reference_filename_geom))
    height, width, _ = im_ref_geom.shape
    im_ref_gray = cv2.cvtColor(im_ref_geom, cv2.COLOR_BGR2GRAY)
    clahe_geom = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_ref_gray = clahe_geom.apply(im_ref_gray)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im_ref_gray, None)

    if des1 is None:
        print("Impossible de trouver des descripteurs dans l'image de référence géométrique.")
        return

    # --- PASSE 2: Traitement de chaque image ---
    print("\n--- Passe 2/2: Alignement et normalisation de chaque image...")
    filenames, aligned_images, valid_masks = [], [], []
    for filename in all_image_files:
        print(f"\nTraitement de l'image : {filename}...")
        im_to_process = cv2.imread(os.path.join(input_dir, filename))
        if im_to_process is None: continue

        # 1. Normalisation de la couleur vers la moyenne globale
        #    (sert toujours au calcul de l'alignement, quel que soit le mode)
        center, radius = find_retina_circle(im_to_process)
        if center is None: continue
        process_mask = np.zeros(im_to_process.shape[:2], dtype="uint8")
        cv2.circle(process_mask, center, radius, 255, -1)
        im_color_corrected = color_transfer_to_target(im_to_process, process_mask, target_stats)

        # 2. Alignement géométrique
        im_to_align_gray = cv2.cvtColor(im_color_corrected, cv2.COLOR_BGR2GRAY)
        im_to_align_gray = clahe_geom.apply(im_to_align_gray)
        kp2, des2 = sift.detectAndCompute(im_to_align_gray, None)
        if des2 is None: continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]

        # En mode 'reference', on recale l'image d'origine : la couleur est corrigée après
        im_source = im_color_corrected if color_mode == 'global' else im_to_process
        im_aligned = im_source # Garder l'image si l'alignement échoue
        valid_mask = process_mask
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None:
                im_aligned = cv2.warpPerspective(im_source, M, (width, height))
                valid_mask = cv2.warpPerspective(process_mask, M, (width, height), flags=cv2.INTER_NEAREST)

        filenames.append(filename)
        aligned_images.append(im_aligned)
        valid_masks.append(valid_mask)

    # Masque final commun : un cercle parfait, identique pour toutes les dates
    common_mask = common_circle_mask(valid_masks)

    # 3. Uniformisation de la colorimétrie par rapport à la référence temporelle
    if color_mode == 'reference':
        print("\n--- Uniformisation de la colorimétrie (référence médiane)...")
        normalize_to_reference(aligned_images, valid_masks, illum_degree, hist_match)

    # 4. Masquage, finition CLAHE éventuelle et sauvegarde
    for filename, im_aligned in zip(filenames, aligned_images):
        if clahe:
            final_image = apply_clahe(im_aligned, common_mask)
        else:
            final_image = cv2.bitwise_and(im_aligned, im_aligned, mask=common_mask)

        output_path = os.path.join(output_dir, f"aligned_{filename}")
        cv2.imwrite(output_path, final_image)
        print(f"  -> Image sauvegardée : {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script pour aligner et normaliser des images de rétine.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images à aligner.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images alignées.")
    parser.add_argument('--color_mode', choices=['reference', 'global'], default='reference',
                        help="'reference' (défaut) : correction locale de l'éclairage vers la médiane de toutes les dates. "
                             "'global' : ancienne méthode (moyenne globale + CLAHE).")
    parser.add_argument('--illum_degree', type=int, default=3,
                        help="Degré de la surface polynomiale modélisant l'éclairage en mode 'reference' (défaut : 3). "
                             "Plus élevé = corrige des ombres plus complexes, mais risque d'atténuer de vraies évolutions.")
    parser.add_argument('--hist_match', action='store_true',
                        help="Active un histogram matching final en mode 'reference'. Déconseillé pour le suivi : "
                             "il impose la même distribution de couleurs à toutes les dates et atténue l'atrophie.")
    parser.add_argument('--clahe', action=argparse.BooleanOptionalAction, default=None,
                        help="Force (--clahe) ou désactive (--no-clahe) le CLAHE final. "
                             "Par défaut : actif en mode 'global', inactif en mode 'reference'.")
    args = parser.parse_args()
    align_images(args.input_dir, args.output_dir, args.color_mode, args.illum_degree,
                 args.hist_match, args.clahe)
