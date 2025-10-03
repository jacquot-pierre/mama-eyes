import cv2
import numpy as np
import os
import argparse
import re
from datetime import datetime

def preprocess_for_segmentation(image_gray):
    """ Prépare l'image pour une segmentation optimale des druses. """
    # 1. Améliorer le contraste local pour faire ressortir les druses
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image_gray)

    # 2. Lisser l'image pour réduire le bruit tout en préservant les bords
    blurred = cv2.medianBlur(contrast_enhanced, 5)
    
    return blurred

def analyze_drusen_regression_by_segmentation(input_dir, output_dir, block_size=151, C=-8):
    """
    Analyse la régression des druses en utilisant une segmentation pré-traitée.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(image_files) < 2: return
    except FileNotFoundError: return

    image_data = []
    for filename in image_files:
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
        if match:
            date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
            image_data.append({'path': os.path.join(input_dir, filename), 'date': datetime.strptime(date_str, '%Y-%m-%d'), 'filename': filename})
    image_data.sort(key=lambda x: x['date'])

    first_image = cv2.imread(image_data[0]['path'])
    if first_image is None: return
    height, width, _ = first_image.shape
    cumulative_regression_mask = np.zeros((height, width), dtype="uint8")

    print(f"Analyse par segmentation pré-traitée dans : {input_dir}")

    # Traiter la première image pour l'affichage, sans régression
    first_image_data = image_data[0]
    output_image = cv2.imread(first_image_data['path'])
    output_filename = f"preprocessed_regression_up_to_{first_image_data['date'].strftime('%Y%m%d')}.jpg"
    output_path = os.path.join(output_dir, output_filename)
    cv2.imwrite(output_path, output_image)
    print(f"Image initiale sauvegardée : {output_path}")

    # Boucler sur les paires d'images consécutives, en commençant par la première paire
    for i in range(len(image_data) - 1):
        image_A_data = image_data[i]
        image_B_data = image_data[i+1]

        print(f"\nComparaison de {image_A_data['filename']} avec {image_B_data['filename']}...")

        image_A_gray = cv2.cvtColor(cv2.imread(image_A_data['path']), cv2.COLOR_BGR2GRAY)
        image_B_gray = cv2.cvtColor(cv2.imread(image_B_data['path']), cv2.COLOR_BGR2GRAY)

        if image_A_gray is None or image_B_gray is None: continue

        # 1. Pré-traiter chaque image
        preprocessed_A = preprocess_for_segmentation(image_A_gray)
        preprocessed_B = preprocess_for_segmentation(image_B_gray)

        # 2. Appliquer la segmentation adaptative sur les images pré-traitées
        mask_A = cv2.adaptiveThreshold(preprocessed_A, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, block_size, C)
        mask_B = cv2.adaptiveThreshold(preprocessed_B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                       cv2.THRESH_BINARY, block_size, C)

        # 3. Trouver ce qui est dans A mais PAS dans B
        disappeared_mask = cv2.subtract(mask_A, mask_B)

        # 4. Nettoyer le masque de disparition
        kernel = np.ones((3,3), np.uint8)
        cleaned_disappeared_mask = cv2.morphologyEx(disappeared_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        # 5. Ajouter au masque cumulatif
        cumulative_regression_mask = cv2.bitwise_or(cumulative_regression_mask, cleaned_disappeared_mask)

        # 6. Dessiner le résultat
        output_image = cv2.imread(image_B_data['path'])
        overlay = output_image.copy()
        overlay[cumulative_regression_mask > 0] = (0, 255, 0) # Surlignage en vert
        
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

        contours, _ = cv2.findContours(cumulative_regression_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  -> {len(contours)} zones de régression cumulées détectées.")

        # Sauvegarder l'image
        output_filename = f"analysis_{image_B_data['date'].strftime('%Y_%m_%d')}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output_image)
        print(f"  -> Image d'analyse sauvegardée : {output_path}")

    print("\nAnalyse terminée.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyser la régression des druses par segmentation pré-traitée.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images d'analyse.")
    parser.add_argument('--block_size', type=int, default=151, help="Taille du voisinage pour le seuillage adaptatif (doit être impair).")
    parser.add_argument('--C', type=int, default=-8, help="Constante soustraite à la moyenne locale.")
    args = parser.parse_args()
    
    analyze_drusen_regression_by_segmentation(args.input_dir, args.output_dir, args.block_size, args.C)