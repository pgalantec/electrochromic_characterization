import shutil
from pathlib import Path


def copiar_txt(directorio_origen, directorio_destino):
    # Convertimos las rutas a objetos Path
    origen = Path(directorio_origen)
    destino = Path(directorio_destino)

    # Recorre la estructura de carpetas en el directorio de origen
    for archivo_origen in origen.rglob("*.txt"):
        # Calcula la ruta relativa desde el origen
        ruta_relativa = archivo_origen.relative_to(origen)
        archivo_destino = destino / ruta_relativa

        # Crea los directorios en la ruta destino si no existen
        archivo_destino.parent.mkdir(parents=True, exist_ok=True)

        # Copia el archivo .txt
        shutil.copy2(archivo_origen, archivo_destino)
        print(f"Archivo copiado: {archivo_destino}")


# Ejemplo de uso
directorio_origen = "output"
directorio_destino = "output_report"

copiar_txt(directorio_origen, directorio_destino)
