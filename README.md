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


# Description des scripts Python

Ce projet contient trois scripts principaux, chacun ayant un rôle spécifique dans le traitement et l'analyse des images de fond d'œil :

### 1. `align_images.py`
Ce script est responsable de la normalisation des couleurs et de l'alignement géométrique des clichés de fond d'œil pour pouvoir les comparer.
- **Normalisation des couleurs (Passe 1)** : Il parcourt toutes les images pour calculer des statistiques globales de couleur (dans l'espace colorimétrique LAB) à l'intérieur du disque rétinien. Il ajuste ensuite chaque image afin qu'elle corresponde à cette palette moyenne, gommant ainsi les variations dues aux changements d'éclairage.
- **Alignement géométrique (Passe 2)** : Il se sert de la première image comme référence. L'algorithme SIFT détecte les points d'intérêts clés pour calculer une homographie. Cela superpose et redresse automatiquement les images suivantes.
- **Traitement final** : L'algorithme CLAHE est passé sur les images alignées pour améliorer le contraste, ce qui facilite grandement la repérage des druses. Enfin, le bord de la rétine est masqué circulairement.

### 2. `analyze_drusen.py`
Ce script analyse l'évolution et la potentielle disparition (régression) progressive des druses.
- **Comparaison chronologique et soustraction** : L'algorithme se veut robuste, il passe sur des groupes de 3 images chronologiques (A, B, C). 
- **Confirmation de la régression** : La détection est soumise à un seuil : la disparition d'une druse n'est confirmée qui si elle apparaît en même temps entre l'image A et B, *ET* entre l'image A et l'image C. De cette façon, les fausses alertes, flous passagers ou artefacts lumineux sont écartés.
- **Visualisation des améliorations** : Les zones de druses "guéries" ou en voie de réduction sont extraites, stockées dans un "masque cumulatif", et tracées avec un contour vert sur les photos de sortie, pour mieux constater l'évolution au premier coup d'œil.

### 3. `create_video.py`
Ce script va compiler l'historique visuel sous forme de deux vidéos distinctes pour bien se rendre compte de l'évolution.
- **Traitement des métadonnées** : Il récupère les dates exactes dans les noms de fichiers (format YYYY_MM_DD).
- **Timeline vidéo** : Il génère automatiquement une barre de lecture temporelle au bas de l'écran. La vidéo progresse proportionnellement avec les jours (y compris années), et une ligne blanche avance dans la barre.  
- **Vidéos multiples** : Il génère une première vidéo "neutre" (avec les couleurs réelles) et une seconde vidéo "enrichie" où le pourcentage de régression trouvé est directement affiché sur l'image en temps réel.

### 4. L'Interface Graphique (GUI)
Le projet comporte désormais une belle interface graphique développée en `PyQt6` pour s'affranchir des lignes de commande Windows et Linux. L'application `gui_app.py` permet de désigner visuellement un dossier d'entrée et un dossier de sortie, et propose 4 boutons interactifs :
1. L'alignement seul.
2. L'analyse seule.
3. La création de vidéo seule.
4. **"Tout exécuter"** : qui enchaîne automatiquement la chaîne de création complète en un seul clic, avec affichage d'une barre de chargement et d'une fenêtre de log d'erreurs en dessous.

# Exécution des scripts

## Lancement via l'interface Graphique (Recommandé)
Pour utiliser le programme simplifié avec son interface :
```bash
poetry run python gui_app.py
```

## Lancement manuel en ligne de commande (Avancé)
Pour lancer les scripts séparément, vous pouvez exécuter ces différentes commandes :
```bash
poetry run python align_images.py --input_dir ./images_to_align/oeil_droit --output_dir ./aligned_images/oeil_droit

poetry run python align_images.py --input_dir ./images_to_align/oeil_gauche --output_dir ./aligned_images/oeil_gauche

poetry run python create_video.py --input_dir ./aligned_images/oeil_droit --output_file ./oeil_droit.mp4

poetry run python create_video.py --input_dir ./aligned_images/oeil_gauche --output_file ./oeil_gauche.mp4

poetry run python analyze_drusen.py --input_dir ./aligned_images/oeil_droit --output_dir ./analysis_results/oeil_droit --block_size 151 --C "-8"

poetry run python analyze_drusen.py --input_dir ./aligned_images/oeil_gauche --output_dir ./analysis_results/oeil_gauche --block_size 151 --C "-8"
```