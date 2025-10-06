import cv2
import os
import argparse
import re
from datetime import datetime

def create_video_from_images(input_dir, output_file, fps=2):
    """
    Crée une vidéo à partir d'une série d'images triées, en y ajoutant une timeline.
    """
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    # 1. Extraire les dates et les chemins de fichiers
    image_data = []
    for filename in image_files:
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
        if match:
            try:
                date_str = f"{match.group(1)}-{match.group(2)}-{match.group(3)}"
                date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                image_data.append({'path': os.path.join(input_dir, filename), 'date': date_obj})
            except ValueError:
                print(f"  -> Format de date non valide pour {filename}, image ignorée.")
                continue
    
    if not image_data:
        print("Aucune image avec un format de date valide n'a été trouvée.")
        return

    image_data.sort(key=lambda x: x['date'])

    # 2. Préparer la timeline
    min_date = image_data[0]['date']
    max_date = image_data[-1]['date']
    total_days = (max_date - min_date).days
    if total_days == 0: total_days = 1 # Éviter la division par zéro

    # Lire la première image pour obtenir les dimensions
    frame = cv2.imread(image_data[0]['path'])
    if frame is None: return
    height, width, _ = frame.shape

    # Définir le codec et créer l'objet VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"Création de la vidéo : {output_file} à {fps} images par seconde...")

    # 3. Boucler sur les images et générer les trames
    for data in image_data:
        frame = cv2.imread(data['path'])
        if frame is None: continue

        # --- Dessiner la timeline ---
        timeline_y = height - 80
        timeline_start_x = 50
        timeline_end_x = width - 50
        timeline_width = timeline_end_x - timeline_start_x

        # Fond de la timeline
        cv2.rectangle(frame, (timeline_start_x, timeline_y - 10), (timeline_end_x, timeline_y + 10), (50, 50, 50), -1)

        # Progression
        days_from_start = (data['date'] - min_date).days
        progress_width = int((days_from_start / total_days) * timeline_width)
        cv2.rectangle(frame, (timeline_start_x, timeline_y - 10), (timeline_start_x + progress_width, timeline_y + 10), (255, 255, 255), -1)

        # Marqueurs pour les années
        for year in range(min_date.year, max_date.year + 2):
            try:
                year_date = datetime(year, 1, 1)
                if min_date <= year_date <= max_date:
                    days_from_start = (year_date - min_date).days
                    marker_x = timeline_start_x + int((days_from_start / total_days) * timeline_width)
                    cv2.line(frame, (marker_x, timeline_y - 20), (marker_x, timeline_y + 20), (200, 200, 200), 2)
                    cv2.putText(frame, str(year), (marker_x - 40, timeline_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except ValueError: # Gérer les années bissextiles ou autres erreurs de date
                continue

        # Affichage de la date courante
        date_str_display = data['date'].strftime('%Y-%m-%d')
        cv2.putText(frame, date_str_display, (50, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)

        video.write(frame)
    # Rajouté par FJ *******************************************************
    # on répete 10 fois la dernière image pour pouvoir apprécier le résultat final qui est le plus important
    for i in range(10):
        video.write(frame)   
    # *********************************************************************
    video.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée avec succès : {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Créer une vidéo à partir d'images alignées avec une timeline.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--output_file', type=str, required=True, help="Fichier vidéo de sortie (ex: video.mp4).")
    parser.add_argument('--fps', type=int, default=2, help="Images par seconde pour la vidéo (par défaut: 2).")
    
    args = parser.parse_args()
    
    create_video_from_images(args.input_dir, args.output_file, args.fps)
