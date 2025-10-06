import os
import shutil
from datetime import datetime
import cv2
import numpy as np
import re

# --- Fonction 1 : Remplacer les espaces par des underscores ---
def remplacer_espaces(repertoire):
    for racine, dossiers, fichiers in os.walk(repertoire, topdown=False):
        for nom in fichiers:
            if ' ' in nom:
                ancien = os.path.join(racine, nom)
                nouveau = os.path.join(racine, nom.replace(' ', '_'))
                os.rename(ancien, nouveau)
        for nom in dossiers:
            if ' ' in nom:
                ancien = os.path.join(racine, nom)
                nouveau = os.path.join(racine, nom.replace(' ', '_'))
                os.rename(ancien, nouveau)

# --- Fonction 2 : Vider un répertoire ---
def vider_repertoire(repertoire):
    if not os.path.isdir(repertoire):
        print(f"Le répertoire {repertoire} n'existe pas.")
        return
    for item in os.listdir(repertoire):
        chemin = os.path.join(repertoire, item)
        try:
            if os.path.isfile(chemin) or os.path.islink(chemin):
                os.unlink(chemin)
            elif os.path.isdir(chemin):
                shutil.rmtree(chemin)
        except Exception as e:
            print(f"Erreur lors de la suppression de {chemin}: {e}")
    print(f"Contenu de {repertoire} effacé.")

# --- Fonction 3 : Alignement des images (extrait de align_images.py) ---
def find_retina_circle(image):
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
    source_lab = cv2.cvtColor(source_image, cv2.COLOR_BGR2LAB)
    source_l, source_a, source_b = cv2.split(source_lab)
    source_l_mean, source_l_std = cv2.meanStdDev(source_l, mask=source_mask)
    source_a_mean, source_a_std = cv2.meanStdDev(source_a, mask=source_mask)
    source_b_mean, source_b_std = cv2.meanStdDev(source_b, mask=source_mask)
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
    os.makedirs(output_dir, exist_ok=True)
    try:
        all_image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not all_image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
            return
    except FileNotFoundError:
        print(f"Le dossier d'entrée n'existe pas : {input_dir}")
        return
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
    ref_center, ref_radius = find_retina_circle(im_ref_geom)
    standard_mask = np.zeros(im_ref_geom.shape[:2], dtype="uint8")
    cv2.circle(standard_mask, ref_center, ref_radius, 255, -1)
    print("\n--- Passe 2/2: Alignement et normalisation de chaque image...")
    for filename in all_image_files:
        print(f"\nTraitement de l'image : {filename}...")
        im_to_process = cv2.imread(os.path.join(input_dir, filename))
        if im_to_process is None: continue
        center, radius = find_retina_circle(im_to_process)
        if center is None: continue
        process_mask = np.zeros(im_to_process.shape[:2], dtype="uint8")
        cv2.circle(process_mask, center, radius, 255, -1)
        im_color_corrected = color_transfer_to_target(im_to_process, process_mask, target_stats)
        im_to_align_gray = cv2.cvtColor(im_color_corrected, cv2.COLOR_BGR2GRAY)
        im_to_align_gray = clahe_geom.apply(im_to_align_gray)
        kp2, des2 = sift.detectAndCompute(im_to_align_gray, None)
        if des2 is None: continue
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
        im_aligned = im_color_corrected
        if len(good_matches) > 10:
            src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
            M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
            if M is not None:
                im_aligned = cv2.warpPerspective(im_color_corrected, M, (width, height))
        im_masked = cv2.bitwise_and(im_aligned, im_aligned, mask=standard_mask)
        lab = cv2.cvtColor(im_masked, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe_final = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_clahe = clahe_final.apply(l)
        lab_clahe = cv2.merge((l_clahe, a, b))
        final_image = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        final_image = cv2.bitwise_and(final_image, final_image, mask=standard_mask)
        output_path = os.path.join(output_dir, f"aligned_{filename}")
        cv2.imwrite(output_path, final_image)
        print(f"  -> Image sauvegardée : {output_path}")

# --- Fonction 4 : Analyse des druses (extrait de analyze_drusen.py) ---
def preprocess_for_segmentation(image_gray):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    contrast_enhanced = clahe.apply(image_gray)
    blurred = cv2.medianBlur(contrast_enhanced, 5)
    return blurred

def analyze_drusen_regression_by_segmentation(input_dir, output_dir, block_size=151, C=-8):
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
            from datetime import datetime
            image_data.append({'path': os.path.join(input_dir, filename), 'date': datetime.strptime(date_str, '%Y-%m-%d'), 'filename': filename})
    image_data.sort(key=lambda x: x['date'])
    first_image_data = image_data[0]
    first_image = cv2.imread(first_image_data['path'])
    if first_image is None: return
    height, width, _ = first_image.shape
    first_image_gray = cv2.cvtColor(first_image, cv2.COLOR_BGR2GRAY)
    preprocessed_first = preprocess_for_segmentation(first_image_gray)
    initial_drusen_mask = cv2.adaptiveThreshold(preprocessed_first, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, block_size, C)
    initial_drusen_area = cv2.countNonZero(initial_drusen_mask)
    if initial_drusen_area == 0:
        print("Aucune druse détectée dans l'image initiale.")
        return
    cumulative_regression_mask = np.zeros((height, width), dtype="uint8")
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
    print("\nCréation de l'image de synthèse finale...")
    last_image = cv2.imread(image_data[-1]['path'])
    summary_image = last_image.copy()
    alpha = 0.5
    initial_overlay = summary_image.copy()
    initial_overlay[initial_drusen_mask > 0] = (0, 0, 255)
    cv2.addWeighted(initial_overlay, alpha, summary_image, 1 - alpha, 0, summary_image)
    regression_overlay = summary_image.copy()
    regression_overlay[cumulative_regression_mask > 0] = (0, 255, 0)
    cv2.addWeighted(regression_overlay, alpha, summary_image, 1 - alpha, 0, summary_image)
    summary_filename = "final_summary.jpg"
    summary_path = os.path.join(output_dir, summary_filename)
    cv2.imwrite(summary_path, summary_image)
    print(f"  -> Image de synthèse sauvegardée : {summary_path}")
    print("\nAnalyse terminée.")

# --- Fonction 5 : Création de vidéo (extrait de create_video.py) ---
def create_video_from_images(input_dir, output_file, fps=2):
    try:
        image_files = sorted([f for f in os.listdir(input_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
        if not image_files:
            print(f"Aucune image trouvée dans le dossier : {input_dir}")
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
                image_data.append({'path': os.path.join(input_dir, filename), 'date': date_obj})
            except ValueError:
                print(f"  -> Format de date non valide pour {filename}, image ignorée.")
                continue
    if not image_data:
        print("Aucune image avec un format de date valide n'a été trouvée.")
        return
    image_data.sort(key=lambda x: x['date'])
    min_date = image_data[0]['date']
    max_date = image_data[-1]['date']
    total_days = (max_date - min_date).days
    if total_days == 0: total_days = 1
    frame = cv2.imread(image_data[0]['path'])
    if frame is None: return
    height, width, _ = frame.shape
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(output_file, fourcc, fps, (width, height))
    print(f"Création de la vidéo : {output_file} à {fps} images par seconde...")
    for data in image_data:
        frame = cv2.imread(data['path'])
        if frame is None: continue
        timeline_y = height - 80
        timeline_start_x = 50
        timeline_end_x = width - 50
        timeline_width = timeline_end_x - timeline_start_x
        cv2.rectangle(frame, (timeline_start_x, timeline_y - 10), (timeline_end_x, timeline_y + 10), (50, 50, 50), -1)
        days_from_start = (data['date'] - min_date).days
        progress_width = int((days_from_start / total_days) * timeline_width)
        cv2.rectangle(frame, (timeline_start_x, timeline_y - 10), (timeline_start_x + progress_width, timeline_y + 10), (255, 255, 255), -1)
        for year in range(min_date.year, max_date.year + 2):
            try:
                year_date = datetime(year, 1, 1)
                if min_date <= year_date <= max_date:
                    days_from_start = (year_date - min_date).days
                    marker_x = timeline_start_x + int((days_from_start / total_days) * timeline_width)
                    cv2.line(frame, (marker_x, timeline_y - 20), (marker_x, timeline_y + 20), (200, 200, 200), 2)
                    cv2.putText(frame, str(year), (marker_x - 40, timeline_y + 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            except ValueError:
                continue
        date_str_display = data['date'].strftime('%Y-%m-%d')
        cv2.putText(frame, date_str_display, (50, height - 150), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 3, cv2.LINE_AA)
        video.write(frame)
    for i in range(10):
        video.write(frame)
    video.release()
    cv2.destroyAllWindows()
    print(f"Vidéo sauvegardée avec succès : {output_file}")

# --- Pipeline principal ---
def pipeline():
    # 1. Nettoyage des espaces dans les noms
    remplacer_espaces('./Retinos/OD')
    remplacer_espaces('./Retinos/OG')
    # 2. Vidage des répertoires de sortie
    vider_repertoire('./Retinos_aligne/OD')
    vider_repertoire('./Retinos_aligne/OG')
    vider_repertoire('./analysis_results/oeil_droit')
    vider_repertoire('./analysis_results/oeil_gauche')
    # 3. Traitement oeil droit
    align_images('./Retinos/OD/', './Retinos_aligne/OD')
    analyze_drusen_regression_by_segmentation('./Retinos_aligne/OD', './analysis_results/oeil_droit', block_size=151, C=-8)
    date_str = datetime.now().strftime('%Y-%m-%d')
    create_video_from_images('./Retinos_aligne/OD', f'OD-{date_str}.mp4')
    create_video_from_images('./analysis_results/oeil_droit', f'OD_Analysis-{date_str}.mp4')
    # 4. Traitement oeil gauche
    align_images('./Retinos/OG/', './Retinos_aligne/OG')
    analyze_drusen_regression_by_segmentation('./Retinos_aligne/OG', './analysis_results/oeil_gauche', block_size=151, C=-8)
    create_video_from_images('./Retinos_aligne/OG', f'OG-{date_str}.mp4')
    create_video_from_images('./analysis_results/oeil_gauche', f'OG_Analysis-{date_str}.mp4')

if __name__ == '__main__':
    pipeline()
