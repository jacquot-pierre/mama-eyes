import cv2
import numpy as np
import os
import argparse
import re
from datetime import datetime

def preprocess_for_segmentation(image_gray):
    """ Prépare l'image pour une segmentation optimale des druses. """
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image_gray)
    blurred = cv2.medianBlur(contrast_enhanced, 5)
    return blurred

def analyze_drusen_regression_by_segmentation(input_dir, output_dir, block_size=151, C=-8):
    """
    Analyse la régression des druses et calcule le pourcentage de druses initiales disparues.
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

    # 1. Analyser les druses sur la première image pour obtenir la surface de référence
    first_image_data = image_data[0]
    first_image = cv2.imread(first_image_data['path'])
    if first_image is None: return
    height, width, _ = first_image.shape

    first_image_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    preprocessed_first = preprocess_for_segmentation(first_image_gray)
    initial_drusen_mask = cv2.adaptiveThreshold(preprocessed_first, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, \
                                                cv2.THRESH_BINARY, block_size, C)
    initial_drusen_area = cv2.countNonZero(initial_drusen_mask)
    if initial_drusen_area == 0:
        print("Aucune druse détectée dans l'image initiale.")
        return

    cumulative_regression_mask = np.zeros((height, width), dtype="uint8")

    # Sauvegarder la première image avec 0% de régression
    percentage_text_first = "Regression: 0.00%"
    cv2.putText(first_image, percentage_text_first, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
    output_filename_first = f"analysis_{first_image_data['date'].strftime('%Y_%m_%d')}.jpg"
    cv2.imwrite(os.path.join(output_dir, output_filename_first), first_image)

    for i in range(len(image_data) - 1):
        image_A_data = image_data[i]
        image_B_data = image_data[i+1]

        print(f"\nComparaison de {image_A_data['filename']} avec {image_B_data['filename']}...")

        image_A_gray = cv2.cvtColor(cv2.imread(image_A_data['path']), cv2.COLOR_BGR2GRAY)
        image_B_gray = cv2.cvtColor(cv2.imread(image_B_data['path']), cv2.COLOR_BGR2GRAY)

        if image_A_gray is None or image_B_gray is None: continue

        preprocessed_A = preprocess_for_segmentation(image_A_gray)
        preprocessed_B = preprocess_for_segmentation(image_B_gray)

        mask_A = cv2.adaptiveThreshold(preprocessed_A, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
        mask_B = cv2.adaptiveThreshold(preprocessed_B, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)

        disappeared_mask = cv2.subtract(mask_A, mask_B)
        kernel = np.ones((3,3), np.uint8)
        cleaned_disappeared_mask = cv2.morphologyEx(disappeared_mask, cv2.MORPH_OPEN, kernel, iterations=2)

        cumulative_regression_mask = cv2.bitwise_or(cumulative_regression_mask, cleaned_disappeared_mask)

        # Calcul du pourcentage par rapport aux druses initiales
        regression_area = cv2.countNonZero(cumulative_regression_mask)
        percentage = (regression_area / initial_drusen_area) * 100

        output_image = cv2.imread(image_B_data['path'])
        overlay = output_image.copy()
        overlay[cumulative_regression_mask > 0] = (0, 255, 0)
        
        alpha = 0.4
        cv2.addWeighted(overlay, alpha, output_image, 1 - alpha, 0, output_image)

        percentage_text = f"Regression: {percentage:.2f}%"
        print(f"  -> {percentage_text}")
        cv2.putText(output_image, percentage_text, (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)

        output_filename = f"analysis_{image_B_data['date'].strftime('%Y_%m_%d')}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output_image)
        print(f"  -> Image d'analyse sauvegardée : {output_path}")

    # --- Génération de l'image de synthèse finale ---
    print("\nCréation de l'image de synthèse finale...")
    last_image = cv2.imread(image_data[-1]['path'])
    summary_image = last_image.copy()
    alpha = 0.5 # Transparence plus marquée pour la lisibilité

    # Superposer les druses initiales en rouge
    initial_overlay = summary_image.copy()
    initial_overlay[initial_drusen_mask > 0] = (0, 0, 255) # Rouge
    cv2.addWeighted(initial_overlay, alpha, summary_image, 1 - alpha, 0, summary_image)

    # Superposer les régressions en vert
    regression_overlay = summary_image.copy()
    regression_overlay[cumulative_regression_mask > 0] = (0, 255, 0) # Vert
    cv2.addWeighted(regression_overlay, alpha, summary_image, 1 - alpha, 0, summary_image)

    summary_filename = "final_summary.jpg"
    summary_path = os.path.join(output_dir, summary_filename)
    cv2.imwrite(summary_path, summary_image)
    print(f"  -> Image de synthèse sauvegardée : {summary_path}")

    print("\nAnalyse terminée.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyser la régression des druses par segmentation pré-traitée.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images d'analyse.")
    parser.add_argument('--block_size', type=int, default=151, help="Taille du voisinage pour le seuillage adaptatif (doit être impair).")
    parser.add_argument('--C', type=int, default=-8, help="Constante soustraite à la moyenne locale.")
    args = parser.parse_args()
    analyze_drusen_regression_by_segmentation(args.input_dir, args.output_dir, args.block_size, args.C)
