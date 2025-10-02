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
    
    # Utiliser une méthode de seuillage plus robuste
    _, thresh = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Trouver les contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None, None

    # Trouver le plus grand contour
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Trouver le cercle englobant minimum
    (x, y), radius = cv2.minEnclosingCircle(largest_contour)

    # Réduire légèrement le rayon pour être plus conservateur
    radius = radius * 0.95
    
    return (int(x), int(y)), int(radius)

def reinhard_color_transfer(source_image, reference_image, source_mask, reference_mask):
    """
    Normalise la couleur de l'image source en se basant sur la méthode de Reinhard et al.
    en utilisant des masques pour ne considérer que la rétine.
    """
    # Convertir les images de BGR vers LAB
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    # Séparer les canaux L, A, B pour les deux images
    source_l, source_a, source_b = cv2.split(source_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)

    # Calculer la moyenne et l'écart-type pour chaque canal de l'image source, en utilisant le masque
    source_l_mean, source_l_std = cv2.meanStdDev(source_l, mask=source_mask)
    source_a_mean, source_a_std = cv2.meanStdDev(source_a, mask=source_mask)
    source_b_mean, source_b_std = cv2.meanStdDev(source_b, mask=source_mask)

    # Calculer la moyenne et l'écart-type pour chaque canal de l'image de référence, en utilisant le masque
    ref_l_mean, ref_l_std = cv2.meanStdDev(ref_l, mask=reference_mask)
    ref_a_mean, ref_a_std = cv2.meanStdDev(ref_a, mask=reference_mask)
    ref_b_mean, ref_b_std = cv2.meanStdDev(ref_b, mask=reference_mask)

    # Normaliser chaque canal de l'image source
    l_normalized = (source_l - source_l_mean) * (ref_l_std / source_l_std) + ref_l_mean
    a_normalized = (source_a - source_a_mean) * (ref_a_std / source_a_std) + ref_a_mean
    b_normalized = (source_b - source_b_mean) * (ref_b_std / source_b_std) + ref_b_mean

    # S'assurer que les valeurs restent dans les bornes valides (0-255)
    l_normalized = np.clip(l_normalized, 0, 255)
    a_normalized = np.clip(a_normalized, 0, 255)
    b_normalized = np.clip(b_normalized, 0, 255)

    # Fusionner les canaux normalisés
    lab_normalized = cv2.merge([l_normalized.astype('uint8'), a_normalized.astype('uint8'), b_normalized.astype('uint8')])

    # Reconvertir l'image de LAB vers BGR
    bgr_normalized = cv2.cvtColor(lab_normalized, cv2.COLOR_LAB2BGR)

    return bgr_normalized

def align_images(input_dir, output_dir, reference_image_path=None):
    """
    Aligne toutes les images d'un dossier de référence sur une image de référence (spécifiée ou la première).
    """
    # S'assurer que le dossier de sortie existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtenir la liste des images
    try:
        all_image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    # Déterminer l'image de référence
    if reference_image_path:
        if not os.path.isfile(reference_image_path):
            print(f"L'image de référence spécifiée n'existe pas : {reference_image_path}")
            return
        reference_filename = os.path.basename(reference_image_path)
    else:
        reference_filename = all_image_files[0]
        reference_image_path = os.path.join(input_dir, reference_filename)

    print(f"Image de référence : {reference_filename}")
    im_ref = cv2.imread(reference_image_path)
    if im_ref is None:
        print(f"Erreur lors de la lecture de l'image de référence : {reference_image_path}")
        return
    
    height, width, _ = im_ref.shape

    # Traitement de l'image de référence
    ref_center, ref_radius = find_retina_circle(im_ref)
    if ref_center is None:
        print("Impossible de trouver le cercle de la rétine dans l'image de référence.")
        return
    ref_mask = np.zeros(im_ref.shape[:2], dtype="uint8")
    cv2.circle(ref_mask, ref_center, ref_radius, 255, -1)
    standard_mask = ref_mask

    im_ref_standardized = cv2.bitwise_and(im_ref, im_ref, mask=standard_mask)

    # Appliquer aussi CLAHE à l'image de référence pour la cohérence
    lab_ref = cv2.cvtColor(im_ref_standardized, cv2.COLOR_BGR2LAB)
    l_ref, a_ref, b_ref = cv2.split(lab_ref)
    clahe_final_ref = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l_clahe_ref = clahe_final_ref.apply(l_ref)
    lab_clahe_ref = cv2.merge((l_clahe_ref, a_ref, b_ref))
    final_ref_image = cv2.cvtColor(lab_clahe_ref, cv2.COLOR_LAB2BGR)
    final_ref_image = cv2.bitwise_and(final_ref_image, final_ref_image, mask=standard_mask)

    output_ref_path = os.path.join(output_dir, f"aligned_{reference_filename}")
    cv2.imwrite(output_ref_path, final_ref_image)
    print(f"Image de référence sauvegardée dans : {output_ref_path}")

    # Préparation pour la détection de caractéristiques
    im_ref_gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_ref_gray = clahe.apply(im_ref_gray)
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im_ref_gray, None)

    if des1 is None:
        print("Aucun descripteur trouvé dans l'image de référence. Impossible de continuer.")
        return

    # Boucler sur les autres images
    other_image_files = [f for f in all_image_files if f != reference_filename]
    for filename in other_image_files:
        current_path = os.path.join(input_dir, filename)
        print(f"\nTraitement de l'image : {filename}...")

        im_to_align = cv2.imread(current_path)
        if im_to_align is None: continue
        
        src_center, src_radius = find_retina_circle(im_to_align)
        if src_center is None:
            print("  -> Impossible de trouver le cercle de la rétine, image ignorée.")
            continue
        src_mask = np.zeros(im_to_align.shape[:2], dtype="uint8")
        cv2.circle(src_mask, src_center, src_radius, 255, -1)

        im_to_align = reinhard_color_transfer(im_to_align, im_ref, src_mask, ref_mask)
            
        im_to_align_gray = cv2.cvtColor(im_to_align, cv2.COLOR_BGR2GRAY)
        im_to_align_gray = clahe.apply(im_to_align_gray)
        kp2, des2 = sift.detectAndCompute(im_to_align_gray, None)

        if des2 is None: continue

        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError: continue

        print(f"  -> {len(good_matches)} correspondances trouvées.")

        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                im_aligned = cv2.warpPerspective(im_to_align, M, (width, height))
                im_aligned_standardized = cv2.bitwise_and(im_aligned, im_aligned, mask=standard_mask)

                # Étape finale : appliquer CLAHE pour uniformiser la luminosité interne
                lab = cv2.cvtColor(im_aligned_standardized, cv2.COLOR_BGR2LAB)
                l, a, b = cv2.split(lab)
                clahe_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                l_clahe = clahe_final.apply(l)
                lab_clahe = cv2.merge((l_clahe, a, b))
                final_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

                # S'assurer que le fond reste noir
                final_image = cv2.bitwise_and(final_image, final_image, mask=standard_mask)

                output_path = os.path.join(output_dir, f"aligned_{filename}")
                cv2.imwrite(output_path, final_image)
                print(f"  -> Image alignée et sauvegardée : {output_path}")
            else:
                print("  -> Impossible de calculer la matrice d'homographie. Image ignorée.")
        else:
            print(f"  -> Pas assez de correspondances trouvées ({len(good_matches)}/{MIN_MATCH_COUNT}). Image ignorée.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script pour aligner des images de rétine.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images à aligner.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images alignées.")
    parser.add_argument('--reference_image', type=str, default=None, help="Chemin vers l'image de référence pour l'alignement et la couleur.")
    
    args = parser.parse_args()
    
    align_images(args.input_dir, args.output_dir, args.reference_image)