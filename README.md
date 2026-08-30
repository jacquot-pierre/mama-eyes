# Mama Eyes

Suivi photographique longitudinal du fond d'œil, pour observer l'évolution des
druses sur une série de clichés personnels.

> **Outil de suivi personnel, sans finalité diagnostique.** Les mesures
> produites sont des *indicateurs de tendance* sur une série homogène, pas des
> mesures cliniques. La photographie couleur du fond d'œil est une modalité
> médiocre pour quantifier les druses — la référence est l'OCT.

## Installation

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -e ".[dev]"
```

Python 3.12 ou plus. Les dépendances principales sont OpenCV, scikit-image,
SciPy, PySide6 et pyqtgraph.

## Prise en main

```bash
# 1. Importer une bibliothèque existante (l'œil est déduit du nom du dossier)
.venv/bin/python -m mama_eyes.cli importer oeil_droit oeil_gauche

# 2. Dérouler la chaîne : photométrie, gabarit, recalage
.venv/bin/python -m mama_eyes.cli tout

# 3. Contrôler visuellement le recalage
.venv/bin/python -m mama_eyes.cli planche --etape recale

# 4. Calibrer les seuils sur le contrôle négatif (voir plus bas)
.venv/bin/python -m mama_eyes.cli controle --csv

# 5. Analyser et exporter
.venv/bin/python -m mama_eyes.cli evolution --csv

# 6. Vidéos
.venv/bin/python -m mama_eyes.cli video

# Interface graphique
.venv/bin/python -m mama_eyes.cli interface
```

Toutes les commandes acceptent `--oeil D`, `--oeil G` ou les deux (défaut).

## Le contrôle négatif

C'est le composant le plus important de l'outil, et il faut s'en servir avant
d'interpréter le moindre chiffre.

Sur un intervalle de quelques jours, aucune évolution biologique des druses
n'est possible. **Tout ce que la chaîne détecte entre deux clichés rapprochés
est donc du bruit** — recalage imparfait, variation de flash, mise au point.
La commande `controle` compare toutes les paires séparées de dix jours ou
moins et en tire le plancher de sensibilité :

```
  16 paire(s) a 10 jours ou moins (ecart median 7 j)
  => Un ecart n'est credible qu'au-dela de N objets de changement.
```

Une variation qui reste sous ce plancher n'est pas interprétable. Le rapport
sert aussi de test de non-régression : toute dégradation introduite par un
changement de paramètre s'y voit immédiatement.

## Ce que le contrôle négatif a établi sur la série de référence

Mesuré sur les 105 dates × 2 yeux (2018-02-19 → 2025-09-02), après calibration :

| Grandeur | Résultat |
|---|---|
| Uniformité du champ (avant → après correction) | 8,7 % → 2,3 % |
| Recalage | 210/210 valides, résidu médian **1,0 px**, max 1,8 px |
| Dérive d'échelle introduite par le recalage | 0,998 – 1,006 (aucune) |
| Plancher de bruit, voie **différentielle** | 5 à 8 objets de changement |
| Plancher de bruit, voie **détection + suivi** | **203 à 222 faux positifs par date** |
| Instabilité de segmentation à 3–10 jours | 5 à 10 % |

Deux conséquences pratiques, qui ne se devinaient pas avant de mesurer :

1. **Les comptages d'apparitions et de disparitions ne sont pas
   interprétables sur cette série.** Les druses étant confluentes (~770 objets
   détectés par cliché), l'instabilité de segmentation produit plus
   d'événements « confirmés » que l'évolution réelle. L'outil le détecte
   automatiquement et refuse de les présenter comme un résultat.
2. **La surface totale, elle, sort du bruit** : sur l'œil droit,
   −3,1 %/an (t ≈ −5,5 après correction d'autocorrélation), sans corrélation
   avec la netteté des clichés — laquelle ne dérive pas sur la période. C'est
   la grandeur à suivre.

Le seuil différentiel par défaut (26) vient de ce travail : il donne une
séparation d'environ 30× entre les comparaisons à 3–10 jours (bruit pur) et
celles à 3–7 ans. **À recalibrer avec `controle` après tout changement de la
chaîne amont.**

> Une diminution de la surface des druses ne s'interprète pas seule : dans la
> DMLA, la régression des druses accompagne fréquemment l'évolution vers
> l'atrophie. L'outil mesure, il ne conclut pas.

## Organisation

```
mama_eyes/
  config.py         paramètres (dataclasses figées, empreinte = clé de cache)
  domain.py         objets métier
  stockage/         SQLite, bibliothèque de fichiers, nommage
  pretraitement/    champ, illumination, débruitage, vaisseaux, recalage, repères
  analyse/          druses, différentiel, suivi temporel, contrôle négatif
  video/            timelapse, mise en évidence
  ui/               interface PySide6
  pipeline.py       orchestration + cache
data/               hors dépôt Git
  brut/             immuable après import
  derive/           cache reconstructible (supprimable sans perte)
  reference/        images « saines » nettoyées à la main
  exports/
```

Trois règles structurantes :

- **le brut est immuable** — tout le reste se régénère ;
- **la base ne stocke jamais de pixels** — métadonnées et mesures seulement ;
- **chaque artefact dérivé porte l'empreinte des paramètres qui l'ont produit**,
  ce qui invalide le cache automatiquement quand un réglage change.

## Choix techniques notables

**Correction du vignettage avant tout.** L'image est divisée par son champ
d'illumination lent (σ ≈ 1/8 du rayon du champ). Sans cette étape, la dérive
d'éclairage domine le signal de différence et noie l'évolution réelle. Sur la
série de référence, la non-uniformité passe de 8,7 % à 2,3 %.

**Similarité à 4 degrés de liberté, jamais d'homographie.** Une transformation
projective à 8 DDL dispose d'assez de liberté pour « expliquer » une partie du
changement pathologique par une déformation géométrique, et donc pour
l'effacer du signal qu'on cherche à mesurer.

**Gabarit de groupe.** Le recalage vise une anatomie moyenne obtenue par
itération, et non le premier cliché de la série — qui après sept ans de dérive
en est le pire représentant.

**Aucun rehaussement adaptatif avant une mesure.** CLAHE et les étirements de
contraste par percentiles sont réservés à l'affichage et à la détection de
points caractéristiques. Un rehaussement dépendant du contenu se déplace quand
une lésion apparaît, ce qui corrompt la différence censée la révéler.
L'harmonisation entre deux images reste une opération unique et explicite
(régression robuste `I_b ≈ a·I_a + b` sur les seuls pixels de fond).

**Suivi par objets, pas par masque cumulatif.** Les druses sont appariées entre
dates par affectation optimale, ce qui leur donne une identité stable. La règle
« validé seulement si observé sur deux comparaisons consécutives » s'applique
à ces pistes. Le décompte peut donc *diminuer*, et une détection erronée peut
être rétractée — ce qu'un masque cumulatif, qui ne fait que croître, interdit.
Les pistes survivent à une absence isolée : sans cette mémoire, une druse
manquée sur un seul cliché produirait à la fois une fausse disparition et une
fausse apparition.

**Échecs de recalage jamais silencieux.** Le nombre d'inliers RANSAC et le
résidu RMS sont stockés pour chaque cliché ; en dessous du seuil, la séance est
marquée et signalée dans l'interface. Une seule image mal recalée suffit à
produire un faux pic d'apparitions puis de disparitions.

**Repères anatomiques par clic manuel.** Les détecteurs automatiques échouent
précisément en cas de DMLA : un amas de druses confluentes rivalise de clarté
avec le disque optique, et l'atrophie efface la fovéa sombre censée localiser
la macula. Deux clics par œil, propagés à toute la série par les
transformations de recalage, sont bien plus fiables. La détection automatique
ne sert que de proposition.

## Contrôles visuels

- `planche` produit une mosaïque légendée de toute la série : une image mal
  masquée ou mal normalisée y saute aux yeux.
- Le **comparateur clignotant** de l'interface bascule entre deux dates
  recalées — le seul moyen fiable de juger un recalage à l'œil. Des vaisseaux
  immobiles signalent un bon alignement.
- Le **damier** vérifie la continuité des vaisseaux à travers les cases.

## Tests

```bash
.venv/bin/python -m pytest tests/ -q
```

Les tests portent sur les propriétés qui comptent : le recalage retrouve une
transformation connue au pixel près, la correction de vignettage *réduit* la
non-uniformité, une absence isolée ne valide pas une disparition, et deux
images identiques ne produisent aucune différence.

## Limites

- Les petites druses dures sont proches du plancher de bruit ; les
  pseudodruses réticulées sont pratiquement invisibles en photographie couleur.
- Le nombre détecté dérive avec la mise au point et la transparence des
  milieux autant qu'avec la maladie. Une cataracte qui progresse *imite une
  régression*. L'analyse signale une corrélation forte entre le nombre de
  druses et la netteté, qui doit être vérifiée avant toute interprétation.
- La calibration d'échelle repose sur le diamètre papillaire moyen (1,85 mm),
  qui varie de ±15 % selon les individus : les millimètres affichés sont
  indicatifs et ne valent qu'en interne à une série.
- Les fichiers ne portent pas d'EXIF : la date vient du nom de fichier
  (`AAAA_MM_JJ_[DG].jpg`) ou d'une saisie explicite.
