
import cv2
import numpy as np
import os
import argparse

def reinhard_color_transfer(source_image, reference_image):
    """
    Normalise la couleur de l'image source en se basant sur la méthode de Reinhard et al.
    pour transférer la couleur de l'image de référence.
    """
    # Convertir les images de BGR vers LAB
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    ref_lab = cv2.cvtColor(reference_image, cv2.COLOR_BGR2LAB)

    # Séparer les canaux L, A, B pour les deux images
    source_l, source_a, source_b = cv2.split(source_lab)
    ref_l, ref_a, ref_b = cv2.split(ref_lab)

    # Calculer la moyenne et l'écart-type pour chaque canal de l'image source
    source_l_mean, source_l_std = cv2.meanStdDev(source_l)
    source_a_mean, source_a_std = cv2.meanStdDev(source_a)
    source_b_mean, source_b_std = cv2.meanStdDev(source_b)

    # Calculer la moyenne et l'écart-type pour chaque canal de l'image de référence
    ref_l_mean, ref_l_std = cv2.meanStdDev(ref_l)
    ref_a_mean, ref_a_std = cv2.meanStdDev(ref_a)
    ref_b_mean, ref_b_std = cv2.meanStdDev(ref_b)

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

def align_images(input_dir, output_dir):
    """
    Aligne toutes les images d'un dossier de référence sur la première image (chronologiquement)
    et sauvegarde les résultats dans un dossier de sortie.
    """
    # S'assurer que le dossier de sortie existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtenir la liste des images et les trier par nom (donc par date)
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    # Le premier fichier est notre image de référence
    reference_filename = image_files[0]
    reference_path = os.path.join(input_dir, reference_filename)
    print(f"Image de référence : {reference_filename}")

    im_ref = cv2.imread(reference_path)
    if im_ref is None:
        print(f"Erreur lors de la lecture de l'image de référence : {reference_path}")
        return
    
    im_ref_gray = cv2.cvtColor(im_ref, cv2.COLOR_BGR2GRAY)
    height, width = im_ref_gray.shape

    # Sauvegarder l'image de référence telle quelle dans le dossier de sortie
    output_ref_path = os.path.join(output_dir, f"aligned_{reference_filename}")
    cv2.imwrite(output_ref_path, im_ref)
    print(f"Image de référence sauvegardée dans : {output_ref_path}")

    # Appliquer un filtre pour améliorer le contraste (CLAHE)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    im_ref_gray = clahe.apply(im_ref_gray)

    # Initialiser le détecteur de caractéristiques SIFT
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(im_ref_gray, None)

    if des1 is None:
        print("Aucun descripteur trouvé dans l'image de référence. Impossible de continuer.")
        return

    # Boucler sur les autres images pour les aligner
    for filename in image_files[1:]:
        current_path = os.path.join(input_dir, filename)
        print(f"\nTraitement de l'image : {filename}...")

        im_to_align = cv2.imread(current_path)
        if im_to_align is None:
            print(f"  -> Erreur de lecture, image ignorée.")
            continue
        
        # Harmoniser la couleur de l'image courante avec l'image de référence
        im_to_align = reinhard_color_transfer(im_to_align, im_ref)
            
        im_to_align_gray = cv2.cvtColor(im_to_align, cv2.COLOR_BGR2GRAY)
        im_to_align_gray = clahe.apply(im_to_align_gray)

        # Détecter les caractéristiques et descripteurs
        kp2, des2 = sift.detectAndCompute(im_to_align_gray, None)

        if des2 is None:
            print("  -> Aucun descripteur trouvé dans l'image courante, image ignorée.")
            continue

        # Mise en correspondance des descripteurs avec un Brute-Force Matcher pour SIFT
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)

        # Appliquer le "ratio test" de Lowe pour ne garder que les bons matches
        good_matches = []
        try:
            for m, n in matches:
                if m.distance < 0.75 * n.distance:
                    good_matches.append(m)
        except ValueError:
            print("  -> Erreur lors du matching, image ignorée.")
            continue


        print(f"  -> {len(good_matches)} correspondances trouvées.")

        # Il faut un minimum de correspondances pour calculer la transformation
        MIN_MATCH_COUNT = 10
        if len(good_matches) > MIN_MATCH_COUNT:
            # Extraire les coordonnées des points correspondants
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

            # Calculer la matrice d'homographie
            M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            
            if M is not None:
                # Appliquer la transformation (warping)
                im_aligned = cv2.warpPerspective(im_to_align, M, (width, height))

                # Sauvegarder l'image alignée
                output_path = os.path.join(output_dir, f"aligned_{filename}")
                cv2.imwrite(output_path, im_aligned)
                print(f"  -> Image alignée et sauvegardée : {output_path}")
            else:
                print("  -> Impossible de calculer la matrice d'homographie. Image ignorée.")
        else:
            print(f"  -> Pas assez de correspondances trouvées ({len(good_matches)}/{MIN_MATCH_COUNT}). Image ignorée.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script pour aligner des images de rétine.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images à aligner.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images alignées.")
    
    args = parser.parse_args()
    
    align_images(args.input_dir, args.output_dir)
