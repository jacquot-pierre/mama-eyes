
import cv2
import os
import argparse
import re

def create_video_from_images(input_dir, output_file, fps=2):
    """
    Crée une vidéo à partir d'une série d'images triées, en y ajoutant la date.
    """
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return

    # Lire la première image pour obtenir les dimensions
    first_image_path = os.path.join(input_dir, image_files[0])
    frame = cv2.imread(first_image_path)
    if frame is None:
        print(f"Erreur lors de la lecture de la première image : {first_image_path}")
        return
    height, width, layers = frame.shape

    # Définir le codec et créer l'objet VideoWriter
    # Le codec 'mp4v' est un bon choix pour les fichiers .mp4
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))

    print(f"Création de la vidéo : {output_file} à {fps} images par seconde...")

    # Boucler sur toutes les images
    for filename in image_files:
        image_path = os.path.join(input_dir, filename)
        frame = cv2.imread(image_path)
        if frame is None:
            print(f"  -> Erreur de lecture, image {filename} ignorée.")
            continue

        # Extraire la date du nom de fichier (ex: aligned_2022_08_29_D.JPG -> 2022-08-29)
        match = re.search(r'(\d{4}_\d{2}_\d{2})', filename)
        if match:
            date_str = match.group(1).replace('_', '-')
            # Ajouter le texte de la date sur l'image
            font = cv2.FONT_HERSHEY_SIMPLEX
            position = (50, 100) # Position en (x, y) depuis le coin supérieur gauche
            font_scale = 3
            font_color = (255, 255, 255) # Blanc
            line_type = 2
            cv2.putText(frame, date_str, position, font, font_scale, font_color, line_type, cv2.LINE_AA)

        video.write(frame)

    # Libérer l'objet video
    video.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée avec succès : {output_file}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Créer une vidéo à partir d'images alignées.")
    parser.add_argument('--input_dir', type=str, required=True, help="Dossier contenant les images alignées.")
    parser.add_argument('--output_file', type=str, required=True, help="Fichier vidéo de sortie (ex: video.mp4).")
    parser.add_argument('--fps', type=int, default=2, help="Images par seconde pour la vidéo (par défaut: 2).")
    
    args = parser.parse_args()
    
    create_video_from_images(args.input_dir, args.output_file, args.fps)
