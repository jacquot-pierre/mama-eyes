import cv2
import numpy as np
import os
import argparse

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

def align_images(input_dir, output_dir):
    """
    Aligne et normalise les images en utilisant une moyenne de couleur globale.
    """
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

    # Utiliser le masque de la première image comme standard pour toutes
    ref_center, ref_radius = find_retina_circle(im_ref_geom)
    standard_mask = np.zeros(im_ref_geom.shape[:2], dtype="uint8")
    cv2.circle(standard_mask, ref_center, ref_radius, 255, -1)

    # --- PASSE 2: Traitement de chaque image ---
    print("\n--- Passe 2/2: Alignement et normalisation de chaque image...")
    for filename in all_image_files:
        print(f"\nTraitement de l'image : {filename}...")
        im_to_process = cv2.imread(os.path.join(input_dir, filename))
        if im_to_process is None: continue

        # 1. Normalisation de la couleur vers la moyenne globale
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

        im_aligned = im_color_corrected # Garder l'image si l'alignement échoue
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None:
                im_aligned = cv2.warpPerspective(im_color_corrected, M, (width, height))

        # 3. Masquage et finition CLAHE
        im_masked = cv2.bitwise_and(im_aligned, im_aligned, mask=standard_mask)
        lab = cv2.cvtColor(im_masked, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe_final.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        final_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        final_image = cv2.bitwise_and(final_image, final_image, mask=standard_mask)

        # 4. Sauvegarde
        output_path = os.path.join(output_dir, f"aligned_{filename}")
        cv2.imwrite(output_path, final_image)
        print(f"  -> Image sauvegardée : {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script pour aligner et normaliser des images de rétine.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images à aligner.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images alignées.")
    args = parser.parse_args()
    align_images(args.input_dir, args.output_dir)
