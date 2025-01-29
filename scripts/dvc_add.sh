#!/bin/bash

# Directorio raíz, se asume que se pasa como argumento al script
ROOT_DIR=$1

# Verifica si se proporcionó el directorio raíz
if [ -z "$ROOT_DIR" ]; then
  echo "Include the directory as input"
  exit 1
fi

# Verifica si el directorio existe
if [ ! -d "$ROOT_DIR" ]; then
  exit 1
fi

# Añade el directorio raíz como una entrada en DVC
dvc add "$ROOT_DIR"

# Recorre cada subdirectorio en el directorio raíz
for DIR in "$ROOT_DIR"/*/; do
  if [ -d "$DIR" ]; then
    echo "Creating $DIR .dvc"
    dvc add "$DIR"
  fi
done

echo "Done!"