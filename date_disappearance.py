import cv2
import numpy as np
import os
import argparse
import csv
import re
from datetime import datetime

# Même échelle que detect_drusen.py : densité = valeur du PNG / DENSITY_SCALE
DENSITY_SCALE = 4000

def load_density_series(density_dir, scale):
    """
    Charge les cartes de densité de toutes les dates, réduites d'un facteur `scale`
    (la densité étant déjà lissée sur ~40 px, on ne perd pas d'information utile).
    Retourne les dates (datetime), la pile (dates, hauteur, largeur) et la taille d'origine.
    """
    files = sorted(f for f in os.listdir(density_dir) if f.endswith('.png'))
    dates, maps, full_size = [], [], None
    for f in files:
        match = re.search(r'(\d{4})_(\d{2})_(\d{2})', f)
        if not match:
            continue
        image = cv2.imread(os.path.join(density_dir, f), cv2.IMREAD_GRAYSCALE)
        full_size = (image.shape[1], image.shape[0])
        small = cv2.resize(image, (full_size[0] // scale, full_size[1] // scale), interpolation=cv2.INTER_AREA)
        dates.append(datetime(int(match.group(1)), int(match.group(2)), int(match.group(3))))
        maps.append(small.astype(np.float32) / DENSITY_SCALE)
    return dates, np.stack(maps), full_size

def rolling_median(series, window):
    """Médiane glissante le long des dates (axe 0), bords prolongés."""
    half = window // 2
    padded = np.concatenate([np.repeat(series[:1], half, axis=0), series, np.repeat(series[-1:], half, axis=0)])
    return np.median(np.lib.stride_tricks.sliding_window_view(padded, window, axis=0), axis=-1)

def analyze_series(series, edge_dates, smooth_window):
    """
    Pour chaque point (colonne de `series`, de forme (dates, points)) :
      - niveau de début et de fin : moyenne des `edge_dates` premières / dernières dates ;
      - bruit à court terme : écart-type robuste des variations d'une date à l'autre ;
      - date du changement : sur la courbe lissée, moment où elle franchit pour la dernière
        fois la mi-chemin entre le niveau de début et celui de fin.
    Comparer le début et la fin, plutôt que chercher une marche, fonctionne aussi bien pour
    une chute brutale que pour une baisse progressive ou encore en cours en fin de série.
    Retourne les niveaux de début et de fin, le bruit et l'indice de la date du changement.
    """
    count = series.shape[0]
    start = series[:edge_dates].mean(axis=0)
    end = series[-edge_dates:].mean(axis=0)
    noise = 1.4826 * np.median(np.abs(np.diff(series, axis=0)), axis=0) / np.sqrt(2)
    smooth = rolling_median(series, smooth_window)
    middle = (start + end) / 2
    # Dernière date où la courbe est encore du côté du niveau de début ; le changement est constaté à la suivante
    on_start_side = np.where(start >= end, smooth >= middle, smooth <= middle)
    last_start_side = count - 1 - np.argmax(on_start_side[::-1], axis=0)
    change = np.clip(last_start_side + 1, 0, count - 1)
    return start, end, noise, change

def year_colors(dates):
    """Une couleur par année (dégradé du bleu, plus ancien, au rouge, plus récent)."""
    years = list(range(dates[0].year, dates[-1].year + 1))
    ramp = cv2.applyColorMap(np.linspace(0, 255, len(years)).astype(np.uint8)[:, None], cv2.COLORMAP_TURBO)[:, 0]
    return {year: tuple(int(c) for c in color) for year, color in zip(years, ramp)}

def draw_map(background, year_map, appeared, colors, title):
    """
    Carte des disparitions : chaque zone est colorée selon l'année où ses druses ont
    disparu, sur la photo la plus récente. Les zones où la densité a augmenté sont
    entourées en blanc. Une légende des années est ajoutée en bas.
    """
    overlay = background.copy()
    for year, color in colors.items():
        overlay[year_map == year] = color
    result = cv2.addWeighted(background, 0.4, overlay, 0.6, 0)
    contours, _ = cv2.findContours(appeared.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (255, 255, 255), 3)

    legend = np.zeros((120, result.shape[1], 3), dtype=np.uint8)
    cv2.putText(legend, title, (30, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.3, (255, 255, 255), 3)
    for i, (year, color) in enumerate(colors.items()):
        x = 30 + i * 260
        cv2.rectangle(legend, (x, 70), (x + 40, 105), color, -1)
        cv2.putText(legend, str(year), (x + 50, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    x = 30 + len(colors) * 260
    cv2.rectangle(legend, (x, 70), (x + 40, 105), (255, 255, 255), 3)
    cv2.putText(legend, "hausse", (x + 50, 102), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2)
    return np.vstack([result, legend])

def draw_check_image(first, last, year_map, appeared, colors, retina, first_date, last_date):
    """
    Image de vérification : première photo, dernière photo et carte des disparitions,
    côte à côte et recadrées sur la rétine. Les contours des zones (couleur de l'année
    de disparition) sont aussi tracés sur les deux photos, pour juger si les druses
    présentes au début ont bien disparu à la fin.
    """
    ys, xs = np.nonzero(retina)
    y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1

    def with_outlines(image):
        result = image.copy()
        for year, color in colors.items():
            contours, _ = cv2.findContours((year_map == year).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(result, contours, -1, color, 4)
        contours, _ = cv2.findContours(appeared.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(result, contours, -1, (255, 255, 255), 4)
        return result

    panels = []
    titles = (f"Premiere photo : {first_date:%Y-%m-%d}", f"Derniere photo : {last_date:%Y-%m-%d}",
              "Annee de disparition des druses")
    filled = draw_map(last, year_map, appeared, colors, "")[:last.shape[0]]
    for image, title in zip((with_outlines(first), with_outlines(last), filled), titles):
        panel = image[y0:y1, x0:x1].copy()
        cv2.putText(panel, title, (30, 70), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (255, 255, 255), 5)
        panels.append(panel)
    body = np.hstack(panels)

    legend = np.zeros((110, body.shape[1], 3), dtype=np.uint8)
    for i, (year, color) in enumerate(colors.items()):
        x = 40 + i * 330
        cv2.rectangle(legend, (x, 30), (x + 60, 85), color, -1)
        cv2.putText(legend, str(year), (x + 75, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    x = 40 + len(colors) * 330
    cv2.rectangle(legend, (x, 30), (x + 60, 85), (255, 255, 255), 4)
    cv2.putText(legend, "hausse", (x + 75, 78), cv2.FONT_HERSHEY_SIMPLEX, 1.8, (255, 255, 255), 4)
    return np.vstack([body, legend])

def draw_bar_chart(rows, colors, path):
    """Histogramme de la surface où les druses ont disparu, par année."""
    width, height, margin = 1200, 600, 80
    chart = np.full((height, width, 3), 255, dtype=np.uint8)
    peak = max([r['disparition_pct_zone'] for r in rows] + [1e-6])
    bar = (width - 2 * margin) // max(len(rows), 1)
    for i, r in enumerate(rows):
        h = int((height - 2 * margin - 40) * r['disparition_pct_zone'] / peak)
        x = margin + i * bar
        cv2.rectangle(chart, (x + 10, height - margin - h), (x + bar - 10, height - margin), colors[r['annee']], -1)
        cv2.putText(chart, str(r['annee']), (x + bar // 2 - 35, height - margin + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
        cv2.putText(chart, f"{r['disparition_pct_zone']:.1f} %", (x + bar // 2 - 40, height - margin - h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    cv2.putText(chart, "Surface ou les druses ont disparu, par annee (% de la zone analysee)", (margin, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2)
    cv2.line(chart, (margin, height - margin), (width - margin, height - margin), (0, 0, 0), 2)
    cv2.imwrite(path, chart)

def date_disappearance(drusen_dir, masks_dir, aligned_dir, output_dir, scale=8, edge_dates=8,
                       smooth_window=5, min_drop=0.008, min_relative_drop=0.3, min_significance=3.0):
    os.makedirs(output_dir, exist_ok=True)
    dates, series, full_size = load_density_series(os.path.join(drusen_dir, 'density'), scale)
    if len(dates) < 3 * edge_dates:
        print(f"Pas assez de dates ({len(dates)}) pour dater des disparitions.")
        return
    small_size = (full_size[0] // scale, full_size[1] // scale)
    zone = cv2.imread(os.path.join(masks_dir, 'analysis_zone.png'), cv2.IMREAD_GRAYSCALE)
    zone_small = cv2.resize(zone, small_size, interpolation=cv2.INTER_NEAREST) > 0
    zone_small = cv2.erode(zone_small.astype(np.uint8), np.ones((3, 3), np.uint8)) > 0

    points = series[:, zone_small]
    start, end, noise, k = analyze_series(points, edge_dates, smooth_window)
    drop = start - end
    # La différence de deux moyennes de `edge_dates` dates a un bruit de noise × √(2 / edge_dates)
    significant = np.abs(drop) >= min_significance * np.maximum(noise, 1e-6) * np.sqrt(2 / edge_dates)
    disappeared = significant & (drop >= min_drop) & (drop >= min_relative_drop * start)
    appeared = significant & (-drop >= min_drop) & (-drop >= min_relative_drop * end)
    print(f"{len(dates)} dates, disparition sur {disappeared.mean():.0%} de la zone, hausse sur {appeared.mean():.0%}")

    # Année de la disparition : date à laquelle le changement est constaté
    years = np.array([d.year for d in dates])
    year_small = np.zeros(small_size[::-1], dtype=np.int32)
    year_small[zone_small] = np.where(disappeared, years[k], 0)
    appeared_small = np.zeros(small_size[::-1], dtype=np.uint8)
    appeared_small[zone_small] = appeared
    year_map = cv2.resize(year_small, full_size, interpolation=cv2.INTER_NEAREST)
    appeared_map = cv2.resize(appeared_small, full_size, interpolation=cv2.INTER_NEAREST) > 0
    zone_full = zone > 0
    year_map[~zone_full] = 0
    appeared_map &= zone_full

    # Carte sur la photo la plus récente
    aligned_files = sorted(f for f in os.listdir(aligned_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg')))
    first = cv2.imread(os.path.join(aligned_dir, aligned_files[0]))
    background = cv2.imread(os.path.join(aligned_dir, aligned_files[-1]))
    colors = year_colors(dates)
    title = f"Annee de disparition des druses ({dates[0]:%Y-%m-%d} -> {dates[-1]:%Y-%m-%d})"
    cv2.imwrite(os.path.join(output_dir, 'disappearance_map.jpg'),
                draw_map(background, year_map, appeared_map, colors, title), [cv2.IMWRITE_JPEG_QUALITY, 92])

    # Vérification : première photo, dernière photo et carte côte à côte
    retina = cv2.imread(os.path.join(masks_dir, 'retina.png'), cv2.IMREAD_GRAYSCALE) > 0
    check = draw_check_image(first, background, year_map, appeared_map, colors, retina, dates[0], dates[-1])
    cv2.imwrite(os.path.join(output_dir, 'disappearance_check.jpg'), check, [cv2.IMWRITE_JPEG_QUALITY, 92])
    cv2.imwrite(os.path.join(output_dir, 'disappearance_year.png'),
                np.where(year_map > 0, year_map - 2000, 0).astype(np.uint8))

    # Bilan par année
    zone_area = np.count_nonzero(zone_small)
    rows = []
    for year in colors:
        area = int(np.count_nonzero(disappeared & (years[k] == year)))
        rows.append({'annee': year, 'disparition_pct_zone': round(100 * area / zone_area, 2)})
    with open(os.path.join(output_dir, 'disappearance_by_year.csv'), 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['annee', 'disparition_pct_zone'])
        writer.writeheader()
        writer.writerows(rows)
    draw_bar_chart(rows, colors, os.path.join(output_dir, 'disappearance_by_year.png'))
    for r in rows:
        print(f"  {r['annee']} : {r['disparition_pct_zone']:.1f} % de la zone")
    print(f"Résultats sauvegardés dans : {output_dir}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Date la disparition des druses à partir des cartes de densité.")
    parser.add_argument('--drusen_dir', type=str, required=True, help="Dossier de sortie de detect_drusen.py (pour un œil).")
    parser.add_argument('--masks_dir', type=str, required=True, help="Dossier des calques produits par build_masks.py.")
    parser.add_argument('--aligned_dir', type=str, required=True, help="Dossier des images alignées (fond de la carte).")
    parser.add_argument('--output_dir', type=str, required=True, help="Dossier où sauvegarder les résultats.")
    parser.add_argument('--edge_dates', type=int, default=8,
                        help="Nombre de dates moyennées pour le niveau de début et celui de fin.")
    parser.add_argument('--smooth_window', type=int, default=5,
                        help="Largeur (en dates) de la médiane glissante utilisée pour dater le changement.")
    parser.add_argument('--min_drop', type=float, default=0.008, help="Baisse minimale de densité entre début et fin.")
    parser.add_argument('--min_relative_drop', type=float, default=0.3,
                        help="Baisse minimale, en fraction de la densité de début.")
    parser.add_argument('--min_significance', type=float, default=3.0,
                        help="La baisse doit dépasser ce multiple du bruit de mesure.")
    args = parser.parse_args()
    date_disappearance(args.drusen_dir, args.masks_dir, args.aligned_dir, args.output_dir,
                       edge_dates=args.edge_dates, smooth_window=args.smooth_window, min_drop=args.min_drop,
                       min_relative_drop=args.min_relative_drop, min_significance=args.min_significance)
