import cv2
import numpy as np
import os
import argparse
import re
from datetime import datetime, timedelta

from date_disappearance import year_colors

# Images destinées à create_video.py : chaque fichier garde la date au format yyyy_mm_dd dans son nom.

def retina_box(masks_dir):
    retina = cv2.imread(os.path.join(masks_dir, 'retina.png'), cv2.IMREAD_GRAYSCALE) > 0
    ys, xs = np.nonzero(retina)
    return ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

def date_of(filename):
    match = re.search(r'(\d{4})_(\d{2})_(\d{2})', filename)
    return datetime(int(match.group(1)), int(match.group(2)), int(match.group(3)))

def tint(image, mask, color, strength):
    """Teinte semi-transparente des pixels du masque."""
    result = image.copy()
    result[mask] = ((1 - strength) * image[mask] + strength * np.array(color)).astype(np.uint8)
    return result

def stable_masks(mask_paths, window, min_area):
    """
    Pour chaque date, druses détectées sur la majorité des `window` dates qui l'entourent,
    sans les toutes petites taches : supprime le scintillement d'une image à l'autre.
    Les masques sont lus au fil de l'eau pour n'en garder que `window` en mémoire.
    """
    half = window // 2
    buffer = {}
    for i in range(len(mask_paths)):
        indices = range(max(i - half, 0), min(i + half, len(mask_paths) - 1) + 1)
        for j in indices:
            if j not in buffer:
                buffer[j] = (cv2.imread(mask_paths[j], cv2.IMREAD_GRAYSCALE) > 0).astype(np.uint8)
        for j in [k for k in buffer if k < i - half]:
            del buffer[j]
        votes = sum(buffer[j] for j in indices)
        majority = (votes * 2 > len(indices)).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(majority, connectivity=8)
        keep = stats[:, cv2.CC_STAT_AREA] >= min_area
        keep[0] = False
        yield keep[labels]

def render_drusen_frames(aligned_dir, drusen_dir, masks_dir, output_dir, window=5, min_area=40):
    """
    Vidéo des druses : photo à gauche, mêmes photo avec les druses en teinte cyan à droite.
    """
    os.makedirs(output_dir, exist_ok=True)
    y0, y1, x0, x1 = retina_box(masks_dir)
    images = sorted(f for f in os.listdir(aligned_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    masks = sorted(f for f in os.listdir(os.path.join(drusen_dir, 'masks')) if f.endswith('.png'))
    mask_paths = [os.path.join(drusen_dir, 'masks', f) for f in masks]
    for image_name, drusen in zip(images, stable_masks(mask_paths, window, min_area)):
        image = cv2.imread(os.path.join(aligned_dir, image_name))
        tinted = tint(image, drusen, (255, 255, 0), 0.55)
        frame = np.hstack([image[y0:y1, x0:x1], tinted[y0:y1, x0:x1]])
        cv2.putText(frame, "photo", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        cv2.putText(frame, "druses detectees", (x1 - x0 + 40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        cv2.imwrite(os.path.join(output_dir, f"druses_{date_of(image_name):%Y_%m_%d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Images sauvegardées dans : {output_dir}")

def render_disappearance_frames(aligned_dir, disappearance_dir, masks_dir, output_dir):
    """
    Vidéo des disparitions : photo à gauche ; à droite, chaque zone se colore (couleur de
    l'année) à partir de la date où ses druses ont disparu.
    """
    os.makedirs(output_dir, exist_ok=True)
    y0, y1, x0, x1 = retina_box(masks_dir)
    day_map = cv2.imread(os.path.join(disappearance_dir, 'disappearance_day.png'), cv2.IMREAD_UNCHANGED).astype(np.int32)
    images = sorted(f for f in os.listdir(aligned_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    dates = [date_of(f) for f in images]
    colors = year_colors(dates)
    origin = datetime(2000, 1, 1)
    year_map = np.zeros_like(day_map)
    for day in np.unique(day_map[day_map > 0]):
        year_map[day_map == day] = (origin + timedelta(days=int(day))).year
    for image_name, date in zip(images, dates):
        image = cv2.imread(os.path.join(aligned_dir, image_name))
        today = (date - origin).days
        colored = image.copy()
        for year, color in colors.items():
            colored = tint(colored, (year_map == year) & (day_map > 0) & (day_map <= today), color, 0.5)
        frame = np.hstack([image[y0:y1, x0:x1], colored[y0:y1, x0:x1]])
        cv2.putText(frame, "photo", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        cv2.putText(frame, "zones ou les druses ont disparu", (x1 - x0 + 40, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.5, (255, 255, 255), 6)
        # Légende des années, en haut à droite
        for i, (year, color) in enumerate(colors.items()):
            x = frame.shape[1] - 260
            cv2.rectangle(frame, (x, 140 + i * 70), (x + 50, 185 + i * 70), color, -1)
            cv2.putText(frame, str(year), (x + 65, 182 + i * 70), cv2.FONT_HERSHEY_SIMPLEX, 1.6, (255, 255, 255), 4)
        cv2.imwrite(os.path.join(output_dir, f"disparitions_{date:%Y_%m_%d}.jpg"), frame,
                    [cv2.IMWRITE_JPEG_QUALITY, 90])
    print(f"Images sauvegardées dans : {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Prépare les images des vidéos de druses et de disparitions "
                                                 "(à assembler ensuite avec create_video.py).")
    parser.add_argument('--aligned_dir', type=str, required=True, help="Dossier des images alignées.")
    parser.add_argument('--masks_dir', type=str, required=True, help="Dossier des calques produits par build_masks.py.")
    parser.add_argument('--drusen_dir', type=str, help="Sortie de detect_drusen.py : produit les images des druses.")
    parser.add_argument('--disappearance_dir', type=str,
                        help="Sortie de date_disappearance.py : produit les images des disparitions.")
    parser.add_argument('--output_dir', type=str, required=True,
                        help="Dossier de sortie (un sous-dossier 'druses' et/ou 'disparitions' y est créé).")
    parser.add_argument('--window', type=int, default=5,
                        help="Nombre de dates sur lesquelles une druse doit être majoritairement détectée pour être affichée.")
    parser.add_argument('--min_area', type=int, default=40, help="Surface minimale (px) d'une druse affichée.")
    args = parser.parse_args()
    if args.drusen_dir:
        render_drusen_frames(args.aligned_dir, args.drusen_dir, args.masks_dir,
                             os.path.join(args.output_dir, 'druses'), args.window, args.min_area)
    if args.disappearance_dir:
        render_disappearance_frames(args.aligned_dir, args.disappearance_dir, args.masks_dir,
                                    os.path.join(args.output_dir, 'disparitions'))
