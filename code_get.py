import os

# Ruta de la carpeta que quieres recorrer
folder_path = r"link\src\modules\gen_sql"

# Archivo donde guardarás el resultado
output_file = r"./resultado.txt"

with open(output_file, "w", encoding="utf-8") as out:
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)

        # Verifica si es archivo
        if os.path.isfile(file_path):
            out.write(f"=== Archivo: {filename} ===\n")

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    contenido = f.read()
                out.write(contenido)
            except:
                out.write("[No se pudo leer este archivo]\n")

            out.write("\n\n---------------------------------------\n\n")

print("Proceso terminado. Archivo generado correctamente.")
