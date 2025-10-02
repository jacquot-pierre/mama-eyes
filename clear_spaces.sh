#!/bin/bash

# Script pour remplacer les espaces par des underscores dans les noms de fichiers et dossiers
# du répertoire courant.

echo "Recherche de fichiers et dossiers avec des espaces..."

# La commande 'find' recherche les éléments dans le répertoire courant ('.')
# dont le nom contient un espace. L'option '-depth' traite le contenu des
# dossiers avant les dossiers eux-mêmes pour éviter les problèmes de renommage.
find $1 -depth -name "* *" -print0 | while IFS= read -r -d $'\0' file; do
    # Pour chaque fichier trouvé, on crée le nouveau nom en remplaçant
    # les espaces par des underscores.
    # La syntaxe ${variable//pattern/replacement} est une substitution globale.
    new_file_name="$(dirname "$file")/$(basename "${file// /_}")"

    # On vérifie que le nouveau nom n'existe pas déjà
    if [ -e "$new_file_name" ]; then
        echo "Attention : Le fichier '$new_file_name' existe déjà. '$file' n'a pas été renommé."
    else
        # On renomme le fichier/dossier
        mv -v "$file" "$new_file_name"
    fi
done

echo "Opération terminée."