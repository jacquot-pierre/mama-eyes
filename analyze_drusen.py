import cv2
import numpy as np
import os
import argparse
import re
import json
from datetime import datetime

def analyze_drusen_regression(input_dir, output_dir, threshold=30):
    """
    Analyse la régression des druses de manière cumulative en confirmant
    la disparition sur deux images consécutives.
    Exporte également un fichier JSON des pourcentages de régression.
    """
    os.makedirs(output_dir, exist_ok=True)

    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(image_files) < 3:
            print("Il faut au moins trois images pour effectuer une comparaison robuste.")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    image_data = []
    for filename in image_files:
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
        if match:
            try:
                date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                image_data.append({'path': os.path.join(input_dir, filename), 'date': date_obj, 'filename': filename})
            except ValueError:
                continue
    
    image_data.sort(key=lambda x: x['date'])

    first_image = cv2.imread(image_data[0]['path'])
    if first_image is None: return
    height, width, _ = first_image.shape
    cumulative_mask = np.zeros((height, width), dtype="uint8")

    # Calcul de la surface de référence (le disque rétinien) pour les pourcentages
    first_image_blur = cv2.GaussianBlur(first_image, (0,0), 3)
    first_image_gray = cv2.cvtColor(first_image_blur, cv2.COLOR_BGR2GRAY)
    _, thresh_ref = cv2.threshold(first_image_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    contours_ref, _ = cv2.findContours(thresh_ref, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    reference_area = 1.0 # fallback
    if contours_ref:
        largest_contour_ref = max(contours_ref, key=cv2.contourArea)
        _, radius_ref = cv2.minEnclosingCircle(largest_contour_ref)
        reference_area = np.pi * ((radius_ref * 0.90) ** 2)

    export_data = [] # Pour stocker les données JSON

    print(f"Analyse cumulative robuste de la régression des druses dans : {input_dir}")

    # Boucler sur les triplets d'images (A, B, C)
    for i in range(len(image_data) - 2):
        image_A_data = image_data[i]
        image_B_data = image_data[i+1]
        image_C_data = image_data[i+2]

        print(f"\nComparaison : {image_A_data['filename']} -> {image_B_data['filename']} -> {image_C_data['filename']}...")

        image_A_gray = cv2.cvtColor(cv2.imread(image_A_data['path']), cv2.COLOR_BGR2GRAY)
        image_B_gray = cv2.cvtColor(cv2.imread(image_B_data['path']), cv2.COLOR_BGR2GRAY)
        image_C_gray = cv2.cvtColor(cv2.imread(image_C_data['path']), cv2.COLOR_BGR2GRAY)

        if image_A_gray is None or image_B_gray is None or image_C_gray is None:
            print("  -> Erreur de lecture d'une des images, triplet ignoré.")
            continue

        # Différence entre A et B, et entre A et C
        diff_AB = cv2.subtract(image_A_gray, image_B_gray)
        diff_AC = cv2.subtract(image_A_gray, image_C_gray)

        # Seuil sur les deux différences
        _, thresh_AB = cv2.threshold(diff_AB, threshold, 255, cv2.THRESH_BINARY)
        _, thresh_AC = cv2.threshold(diff_AC, threshold, 255, cv2.THRESH_BINARY)

        # Une régression est confirmée si elle apparaît dans les deux comparaisons
        confirmed_diff = cv2.bitwise_and(thresh_AB, thresh_AC)

        kernel = np.ones((3,3), np.uint8)
        new_regressions = cv2.morphologyEx(confirmed_diff, cv2.MORPH_OPEN, kernel, iterations=1)

        # Ajouter les nouvelles régressions confirmées au masque cumulatif
        cumulative_mask = cv2.bitwise_or(cumulative_mask, new_regressions)

        # Dessiner le masque cumulatif sur l'image la plus récente (image C)
        image_C_color = cv2.imread(image_C_data['path'])
        output_image = image_C_color.copy()
        
        contours, _ = cv2.findContours(cumulative_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  -> {len(contours)} zones de régression cumulées détectées.")
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2)

        # Sauvegarder l'image résultat
        output_filename = f"cumulative_regression_up_to_{image_C_data['date'].strftime('%Y%m%d')}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output_image)

        # Calculer le pourcentage de régression
        regression_pixels = cv2.countNonZero(cumulative_mask)
        percentage = (regression_pixels / reference_area) * 100.0
        print(f"  -> Image d'analyse cumulative sauvegardée : {output_path}")
        print(f"  -> Régression cumulée : {percentage:.4f}% de la surface de référence.")

        # Ajouter aux données d'export JSON
        export_data.append({
            "date": image_C_data['date'].strftime('%Y-%m-%d'),
            "percentage": round(percentage, 4),
            "filename": output_filename
        })

    # Sauvegarde du fichier JSON final
    json_path = os.path.join(output_dir, "evolution_stats.json")
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=4)
        
    print(f"\nAnalyse cumulative terminée. Données exportées vers : {json_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyser la régression des druses de manière cumulative et robuste.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images d'analyse.")
    parser.add_argument('--threshold', type=int, default=25, help="Seuil de détection de changement.")
    
    args = parser.parse_args()
    
    analyze_drusen_regression(args.input_dir, args.output_dir, args.threshold)