# Installation de la librairie
Tout d'abord installer poetry
```bash
pip install poetry
```

Se déplacer dans le dossier mama-eyes et exécuter la commande
```bash
poetry config virtualenvs.in-project true
```

Cela permet de créer le virtualenv python directement dans le dossier mama-eyes et pas de façon globale

Puis installer les dépendances
```bash
poetry install
```


# Préparation des images
Les photos sont placées dans `./images_to_align/oeil_droit` et `./images_to_align/oeil_gauche`
(ces dossiers ne sont pas versionnés). Chaque fichier doit être nommé avec sa date au format
`yyyy_mm_dd.jpg` (ex : `2018_02_19.jpg`) : c'est cette date qui sert à trier les images et à
construire la timeline des vidéos. C'est le dossier qui indique de quel œil il s'agit.


# Exécution des scripts
Pour lancer le script permettant d'aligner les images, exécuter
```bash
poetry run python align_images.py --input_dir ./images_to_align/oeil_droit --output_dir ./aligned_images/oeil_droit

poetry run python align_images.py --input_dir ./images_to_align/oeil_gauche --output_dir ./aligned_images/oeil_gauche

poetry run python create_video.py --input_dir ./aligned_images/oeil_droit --output_file ./oeil_droit.mp4

poetry run python create_video.py --input_dir ./aligned_images/oeil_gauche --output_file ./oeil_gauche.mp4

poetry run python build_masks.py --input_dir ./aligned_images/oeil_droit --output_dir ./masks/oeil_droit

poetry run python build_masks.py --input_dir ./aligned_images/oeil_gauche --output_dir ./masks/oeil_gauche

poetry run python detect_drusen.py --input_dir ./aligned_images/oeil_droit --masks_dir ./masks/oeil_droit --output_dir ./drusen/oeil_droit

poetry run python detect_drusen.py --input_dir ./aligned_images/oeil_gauche --masks_dir ./masks/oeil_gauche --output_dir ./drusen/oeil_gauche

poetry run python create_video.py --input_dir ./drusen/oeil_droit/overlays --output_file ./druses_oeil_droit.mp4

poetry run python create_video.py --input_dir ./drusen/oeil_droit/density_overlays --output_file ./densite_oeil_droit.mp4

poetry run python date_disappearance.py --drusen_dir ./drusen/oeil_droit --masks_dir ./masks/oeil_droit --aligned_dir ./aligned_images/oeil_droit --output_dir ./disappearance/oeil_droit

poetry run python date_disappearance.py --drusen_dir ./drusen/oeil_gauche --masks_dir ./masks/oeil_gauche --aligned_dir ./aligned_images/oeil_gauche --output_dir ./disappearance/oeil_gauche

poetry run python analyze_drusen.py --input_dir ./aligned_images/oeil_droit --output_dir ./analysis_results/oeil_droit --threshold 25

poetry run python analyze_drusen.py --input_dir ./aligned_images/oeil_gauche --output_dir ./analysis_results/oeil_gauche --threshold 25
```


# Calques d'exclusion (`build_masks.py`)
Préparation de la détection des druses : certaines structures sont souvent prises à tort pour
des druses (ou cachent leur évolution) et doivent être exclues de l'analyse. Les images alignées
étant superposables au pixel près, les calques sont calculés une seule fois sur l'image médiane
de toutes les dates (nette et sans défaut propre à une photo), puis valent pour toute la série.

- **Vaisseaux** : structures sombres et fines dans le canal vert, mais presque invisibles dans le
  canal rouge (les vaisseaux sont rouges). Ce critère évite de confondre les vaisseaux avec les
  espaces sombres entre druses voisines, qui eux sont sombres dans les deux canaux.
- **Papille** : disque le plus clair par rapport à son entourage, cherché dans le canal rouge après
  effacement des vaisseaux qui la traversent.
- **Fovéa** : point le plus sombre de la macula, à environ 2,5 diamètres papillaires de la papille.
  Seul le petit centre fovéal est masqué : la macula, où se trouvent la plupart des druses, reste analysée.
- **Bord** : fine couronne au bord du disque rétinien.

Le dossier de sortie contient `reference.png` (image médiane), un PNG par calque, `analysis_zone.png`
(rétine moins tous les calques) et `control.jpg`, la référence avec les calques en couleur pour
vérifier le résultat à l'œil. Options : `--vessel_margin`, `--disc_margin`, `--fovea_ratio`, `--border_width`.


# Détection des druses par date (`detect_drusen.py`)
Les druses sont cherchées sur chaque date, dans la zone d'analyse définie par les calques
(il faut donc lancer `build_masks.py` avant, qui fournit aussi l'image médiane).

- **Fond commun** : le niveau de la rétine « ordinaire » est estimé une seule fois, sur l'image
  médiane de toutes les dates, en ignorant les druses, les zones sombres (atrophie) et les vaisseaux.
  Deux pièges sont ainsi évités :
  - dans les zones presque couvertes de druses, un fond calculé localement serait au niveau des
    druses elles-mêmes, qui ne ressortiraient qu'à peine et seraient détectées une date sur deux ;
  - un fond propre à chaque date suivrait l'évolution de la rétine (atrophie qui s'étend) et ferait
    apparaître de fausses « nouvelles druses » autour des zones qui changent.
  Grâce à l'harmonisation des couleurs entre dates, une druse qui se résorbe redescend simplement
  au niveau de ce fond commun.
- **Score de druse** : contraste relatif du canal vert (vaisseaux effacés) par rapport au fond commun.
- **Médiane sur plusieurs dates** : le score de chaque date est la médiane de celui de
  `--temporal_window` dates consécutives (3 par défaut). Une druse doit donc être vue sur la majorité
  de ces dates : une photo atypique ne crée plus d'apparition ou de disparition isolée.
- **Seuillage à deux niveaux** : une druse est une région de contraste supérieur à `--low` (0,12)
  contenant au moins un point supérieur à `--high` (0,18). Cela élimine le bruit tout en gardant
  une forme fidèle.
- **Filtres de forme** : surface minimale (`--min_area`, 40 px) et allongement maximal
  (`--max_elongation`, 4) pour écarter les liserés clairs le long des vaisseaux.

**Petites druses.** En dehors du centre, la rétine est souvent tapissée de petites druses fines et
serrées (10 à 20 px), si nombreuses qu'elles relèvent le fond commun à leur propre niveau : le
détecteur précédent ne les voit pas. Un second détecteur les compare à leur voisinage immédiat
(top-hat morphologique sur ~20 px) : à cette échelle, les petits creux entre deux druses suffisent
à définir le fond. Même médiane temporelle, seuils `--fine_low` (0,06) / `--fine_high` (0,10),
surface minimale `--fine_min_area` (15 px). Le résultat final est l'union des deux détections.
Ces druses étant à peine au-dessus du bruit de l'image, leurs contours individuels restent
approximatifs : pour les suivre dans le temps, la densité ci-dessous est plus fiable.

**Densité de druses fines.** Pour chaque point, intensité moyenne des petites taches claires sur
une fenêtre d'environ `--density_sigma` px (40). Cette carte ne dépend d'aucun seuil et varie très
peu d'une date à l'autre : c'est la mesure la plus stable pour suivre la disparition des druses.

Sorties dans le dossier choisi :
- `scores/score_<date>.png` : score du détecteur principal (contraste relatif × 1000, sur 8 bits) ;
- `masks/drusen_<date>.png` : druses détectées (union des deux détecteurs) ;
- `overlays/drusen_<date>.jpg` : image avec les contours (vert : druses franches, jaune : petites
  druses), utilisable par `create_video.py` ;
- `density/density_<date>.png` : densité de druses fines (× 4000, sur 8 bits) ;
- `density_overlays/density_<date>.jpg` : densité en carte de chaleur sur la photo (clair = dense,
  violet = peu ou pas de druses), utilisable par `create_video.py` ;
- `drusen_summary.csv` : par date, nombre de druses, surface (en pixels et en % de la zone analysée),
  part des petites druses et densité moyenne.

Résultats sur la série actuelle, entre 2018 et 2025 :
- densité moyenne : de 0,046 à 0,033 (œil droit, −27 %) et de 0,046 à 0,036 (œil gauche, −21 %),
  avec une variation d'une date à l'autre plus de dix fois inférieure à cette baisse ;
- surface de druses : de 35 à 24 % de la zone analysée (œil droit) et de 38 à 30 % (œil gauche) ;
- environ 2 % des pixels changent d'état d'une date à l'autre pour le détecteur principal ; les
  druses disparues forment des zones continues, là où l'atrophie s'est étendue.


# Datation des disparitions (`date_disappearance.py`)
Pour chaque point de la zone analysée, on suit la densité de druses fines (produite par
`detect_drusen.py`) sur toutes les dates (calcul à 1/8 de la résolution, la densité étant déjà
lissée sur ~40 px), et on compare le **niveau du début** et le **niveau de la fin** de la série
(moyennes des `--edge_dates` premières et dernières dates, 8 par défaut). Une disparition est retenue si :
- la baisse est nette : au moins `--min_drop` (0,008) et au moins `--min_relative_drop` (30 %)
  de la densité du début ;
- la baisse dépasse `--min_significance` (3) fois le bruit de mesure (estimé à partir des variations
  d'une date à l'autre) ;
- la **luminosité des druses** (score principal de `detect_drusen.py`, lissé à la même échelle que la
  densité) baisse elle aussi d'au moins `--min_score_relative_drop` (20 %) de sa valeur de début.
  La texture fine diminue en effet aussi quand de petites druses grossissent et fusionnent en plaques
  plus lisses : la zone reste alors aussi claire (les druses sont toujours là), ce que ce critère écarte.

La **date de disparition** est celle où la courbe, lissée par une médiane glissante sur
`--smooth_window` dates (5), passe pour la dernière fois sous la mi-chemin entre le niveau du début
et celui de la fin. Cette approche convient aussi bien à une chute brutale qu'à une baisse
progressive sur plusieurs années ou encore en cours en fin de série (une première version, qui
cherchait une « marche » unique dans la courbe, manquait ces deux derniers cas). Les hausses
significatives de densité (nouvelles druses) sont repérées de la même façon.

Sorties dans le dossier choisi :
- `disappearance_map.jpg` : photo la plus récente, chaque zone colorée selon l'année où ses druses
  ont disparu (zones en hausse entourées en blanc), avec légende ;
- `disappearance_check.jpg` : image de vérification, avec côte à côte la première photo, la dernière
  photo (contours des zones tracés sur les deux, couleur de l'année de disparition) et la carte ;
- `disappearance_year.png` : année de disparition par pixel (valeur = année − 2000, 0 = aucune) ;
- `disappearance_by_year.csv` et `disappearance_by_year.png` : surface où les druses ont disparu,
  par année, en % de la zone analysée.

Résultats sur la série actuelle : disparition sur 30 % de la zone analysée pour l'œil droit (pic en
2021) et 17 % pour l'œil gauche (pic en 2023), aucune hausse significative. Les zones signalées
comme faux positifs lors de la vérification visuelle sont écartées par le critère de luminosité. Les zones
disparues sont continues et progressent vers l'extérieur au fil des années. Sur des points témoins,
la date trouvée correspond à la disparition visible sur les photos.


# Alignement et découpage (`align_images.py`)
Chaque photo est recalée sur la première (SIFT + homographie). D'une visite à l'autre, l'œil est
décalé d'environ 250 px dans le cercle de l'appareil photo : une fois la rétine recalée, les disques
photographiés ne se superposent donc plus. Toutes les images sont découpées avec le même cercle,
le plus grand contenu dans la zone visible sur toutes les dates (rayon d'environ 865 px pour des
photos de 3008×2000). Le disque reste ainsi un cercle parfait et identique d'une date à l'autre.

Une seule photo cadrée très différemment des autres suffit à réduire ce cercle pour toute la série :
le rayon est affiché au lancement du script. Les photos du 2018-02-19 et du 2018-06-14 ont été
retirées pour cette raison.


# Uniformisation de la colorimétrie (`align_images.py`)
L'alignement géométrique est identique quel que soit le mode.
La colorimétrie peut être traitée de deux façons, choisies avec `--color_mode` :

- `reference` (par défaut) : les images étant superposables pixel à pixel, on construit une image de
  référence égale à la médiane pixel par pixel de toutes les dates. Pour chaque image et chaque canal
  couleur, le rapport `image / référence` est modélisé par une surface polynomiale lisse, ajustée de
  façon robuste : les zones qui diffèrent réellement de la référence (druses, atrophie) sont exclues
  de l'ajustement. L'image est ensuite divisée par cette surface. Cela supprime les ombres, le
  vignettage et les dominantes de couleur propres à une prise de vue, tout en conservant les
  évolutions localisées de la rétine.
  Enfin, l'amplitude des détails fins (druses, vaisseaux, texture) est harmonisée d'une date à l'autre :
  elle est mesurée uniquement sur les zones stables dans le temps, puis chaque image est ramenée à
  l'amplitude médiane de la série (photos plus « dures » ou plus « douces »).
- `global` : ancienne méthode, transfert de couleur vers la moyenne/écart-type globaux (Reinhard)
  puis CLAHE sur chaque image.

Options utiles :
- `--illum_degree 3` : degré de la surface polynomiale. Plus élevé, elle corrige des ombres plus
  complexes mais risque d'atténuer de vraies évolutions ; 2 est plus prudent.
- `--no_contrast_norm` : désactive l'harmonisation de l'amplitude des détails.
- `--hist_match` : ajoute un histogram matching final vers l'histogramme commun de toutes les images.
  Déconseillé pour le suivi : en imposant la même distribution de couleurs à toutes les dates, il
  éclaircit les zones d'atrophie qui s'étendent.
- `--clahe` / `--no-clahe` : force ou désactive le CLAHE final (actif par défaut uniquement en mode `global`).
  Le CLAHE rend les images plus contrastées mais réintroduit des écarts d'une date à l'autre.

Pourquoi pas un simple flou pour estimer l'éclairage ? Un flou gaussien (même large) considère
comme « éclairage » toute variation plus grande que sa taille, y compris les plaques d'atrophie :
elles étaient gommées. Une surface polynomiale robuste ne peut pas reproduire une lésion localisée.

Mémoire : toutes les images d'un œil sont gardées en mémoire (environ 4,7 Go de pic pour 103 photos
de 3008×2000).

Précision obtenue sur la série actuelle (103 photos par œil) :
- recalage, mesuré sur les vaisseaux : décalage typique 0,6 px, au pire environ 3 px ;
- couleur : écart moyen d'une image à ses deux voisines dans le temps inférieur à 1 niveau sur 255
  en luminance. Les écarts restants viennent surtout de différences propres à certaines photos
  (mise au point, flash), qu'on ne corrige pas pour ne pas modifier le contenu de la rétine.
