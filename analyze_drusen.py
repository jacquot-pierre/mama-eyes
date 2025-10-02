import cv2
import numpy as np
import os
import argparse
import re
from datetime import datetime

def analyze_drusen_regression(input_dir, output_dir, threshold=30):
    """
    Analyse la régression des druses de manière cumulative et surligne
    l'ensemble des zones de changement au fil du temps.
    """
    # S'assurer que le dossier de sortie existe
    os.makedirs(output_dir, exist_ok=True)

    # Obtenir la liste des images et les trier par date
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if len(image_files) < 2:
            print("Il faut au moins deux images pour effectuer une comparaison.")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    image_data = []
    for filename in image_files:
        # Recherche d'une séquence de date plus flexible
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
        if match:
            try:
                date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                image_data.append({'path': os.path.join(input_dir, filename), 'date': date_obj, 'filename': filename})
            except ValueError:
                continue
    
    image_data.sort(key=lambda x: x['date'])

    # Initialiser un masque cumulatif
    first_image = cv2.imread(image_data[0]['path'])
    if first_image is None:
        print("Erreur de lecture de la première image.")
        return
    height, width, _ = first_image.shape
    cumulative_mask = np.zeros((height, width), dtype="uint8")

    print(f"Analyse cumulative de la régression des druses dans : {input_dir}")

    # Boucler sur les paires d'images consécutives
    for i in range(len(image_data) - 1):
        image_A_data = image_data[i]
        image_B_data = image_data[i+1]

        print(f"\nComparaison de {image_A_data['filename']} avec {image_B_data['filename']}...")

        image_A_gray = cv2.cvtColor(cv2.imread(image_A_data['path']), cv2.COLOR_BGR2GRAY)
        image_B_gray = cv2.cvtColor(cv2.imread(image_B_data['path']), cv2.COLOR_BGR2GRAY)

        if image_A_gray is None or image_B_gray is None:
            print("  -> Erreur de lecture d'une des images, paire ignorée.")
            continue

        # Détecter les nouvelles régressions
        diff = cv2.subtract(image_A_gray, image_B_gray)
        _, thresh = cv2.threshold(diff, threshold, 255, cv2.THRESH_BINARY)
        kernel = np.ones((3,3), np.uint8)
        new_regressions = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel, iterations=1)

        # Ajouter les nouvelles régressions au masque cumulatif
        cumulative_mask = cv2.bitwise_or(cumulative_mask, new_regressions)

        # Dessiner le masque cumulatif sur l'image la plus récente (image B)
        image_B_color = cv2.imread(image_B_data['path'])
        output_image = image_B_color.copy()
        
        # Trouver les contours du masque cumulatif pour les dessiner
        contours, _ = cv2.findContours(cumulative_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        print(f"  -> {len(contours)} zones de régression cumulées détectées.")
        cv2.drawContours(output_image, contours, -1, (0, 255, 0), 2) # Contours en vert

        # Sauvegarder l'image résultat
        output_filename = f"cumulative_regression_up_to_{image_B_data['date'].strftime('%Y%m%d')}.jpg"
        output_path = os.path.join(output_dir, output_filename)
        cv2.imwrite(output_path, output_image)
        print(f"  -> Image d'analyse cumulative sauvegardée : {output_path}")

    print("\nAnalyse cumulative terminée.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Analyser la régression des druses de manière cumulative.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées et traitées.")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les images d'analyse.")
    parser.add_argument('--threshold', type=int, default=30, help="Seuil de détection de changement.")
    
    args = parser.parse_args()
    
    analyze_drusen_regression(args.input_dir, args.output_dir, args.threshold)