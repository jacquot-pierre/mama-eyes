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


# Exécution des scripts
Pour lancer le script permettant d'aligner les images, exécuter
```bash
poetry run python align_images.py --input_dir ./images_to_align/oeil_droit --output_dir ./aligned_images/oeil_droit --reference_image ./images_to_align/oeil_droit/2018_05_14_D.JPG

poetry run python align_images.py --input_dir ./images_to_align/oeil_gauche --output_dir ./aligned_images/oeil_gauche --reference_image ./images_to_align/oeil_gauche/2018_05_25_G.JPG

poetry run python create_video.py --input_dir ./aligned_images/oeil_droit --output_file ./oeil_droit.mp4

poetry run python create_video.py --input_dir ./aligned_images/oeil_gauche --output_file ./oeil_gauche.mp4
```