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

poetry run python analyze_drusen.py --input_dir ./aligned_images/oeil_droit --output_dir ./analysis_results/oeil_droit --threshold 25

poetry run python analyze_drusen.py --input_dir ./aligned_images/oeil_gauche --output_dir ./analysis_results/oeil_gauche --threshold 25
```


# Uniformisation de la colorimétrie (`align_images.py`)
L'alignement géométrique (SIFT + homographie vers la première image) est identique quel que soit le mode.
La colorimétrie peut être traitée de deux façons, choisies avec `--color_mode` :

- `reference` (par défaut) : les images étant superposables pixel à pixel, on construit une image de
  référence égale à la médiane pixel par pixel de toutes les dates. Pour chaque image et chaque canal
  couleur, le rapport `image / référence` est modélisé par une surface polynomiale lisse, ajustée de
  façon robuste : les zones qui diffèrent réellement de la référence (druses, atrophie) sont exclues
  de l'ajustement. L'image est ensuite divisée par cette surface. Cela supprime les ombres, le
  vignettage et les dominantes de couleur propres à une prise de vue, tout en conservant les
  évolutions localisées de la rétine.
- `global` : ancienne méthode, transfert de couleur vers la moyenne/écart-type globaux (Reinhard)
  puis CLAHE sur chaque image.

Options utiles :
- `--illum_degree 3` : degré de la surface polynomiale. Plus élevé, elle corrige des ombres plus
  complexes mais risque d'atténuer de vraies évolutions ; 2 est plus prudent.
- `--hist_match` : ajoute un histogram matching final vers l'histogramme commun de toutes les images.
  Déconseillé pour le suivi : en imposant la même distribution de couleurs à toutes les dates, il
  éclaircit les zones d'atrophie qui s'étendent.
- `--clahe` / `--no-clahe` : force ou désactive le CLAHE final (actif par défaut uniquement en mode `global`).
  Le CLAHE rend les images plus contrastées mais réintroduit des écarts d'une date à l'autre.

Pourquoi pas un simple flou pour estimer l'éclairage ? Un flou gaussien (même large) considère
comme « éclairage » toute variation plus grande que sa taille, y compris les plaques d'atrophie :
elles étaient gommées. Une surface polynomiale robuste ne peut pas reproduire une lésion localisée.

Mémoire : toutes les images d'un œil sont gardées en mémoire (environ 4,5 Go de pic pour 105 photos
de 3008×2000).
