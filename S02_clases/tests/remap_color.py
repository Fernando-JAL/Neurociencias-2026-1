from PIL import Image
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import os


def reescale_map_color(input_path, output_path, min_temperature, max_temperature, cmap_vmin=20.0, cmap_vmax=50.0,
                       legend_path="leyenda_20_50.png"):
    # 1) Abrir imagen y convertir a escala de grises [0, 1]
    img = Image.open(input_path).convert("L")  # "L" = 8-bit grayscale
    gray = np.asarray(img, dtype=np.float32) / 255.0

    # 2) Reescalar a temperaturas físicas en °C usando 20..max_temperature
    #    T = 20 + gray*(maxT - 20)
    temperatures = min_temperature + gray * (max_temperature - min_temperature)

    # 3) Clipping a 20..50 para pintar con el colormap calibrado 20–50 °C
    temperatures_for_colormap = np.clip(temperatures, cmap_vmin, cmap_vmax)

    # 4) Crear el mapa 'hot' con normalización fija 20–50 °C
    norm = mpl.colors.Normalize(vmin=cmap_vmin, vmax=cmap_vmax)
    cmap = mpl.cm.get_cmap("plasma")  # también podrías usar mpl.cm.hot

    # 5) Aplicar colormap -> matriz RGBA en [0,1]
    rgba = cmap(norm(temperatures_for_colormap))  # shape (H, W, 4)

    # 6) Guardar imagen color (RGB en 8 bits)
    rgb_uint8 = (rgba[..., :3] * 255).astype(np.uint8)

    output_folder = "\\".join(output_path.split(sep="\\")[:-1])
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    Image.fromarray(rgb_uint8, mode="RGB").save(output_path)
    print(f"Imagen colorizada guardada en: {output_path}")

    # 7) (Opcional) Guardar una leyenda/escala de colores 20–50 °C
    fig, ax = plt.subplots(figsize=(5, 1.0), dpi=200)
    fig.subplots_adjust(bottom=0.5)

    cb1 = mpl.colorbar.ColorbarBase(
        ax,
        cmap=cmap,
        norm=norm,
        orientation='horizontal'
    )
    cb1.set_label("Temperatura [°C]")
    fig.savefig(legend_path, bbox_inches="tight")
    plt.close(fig)
    print(f"Leyenda guardada en: {legend_path}")

    # 8) Información útil en consola
    print(f"Rango real de la imagen reescalada antes de pintar: "
          f"{temperatures.min():.2f}–{temperatures.max():.2f} °C")
    print(f"El colormap se aplicó con vmin={cmap_vmin} °C y vmax={cmap_vmax} °C.")


# ----------------- Parámetros del usuario -----------------
# input_folder = r"C:\Users\OMEN CI7\Documents\repository\Neurociencias-2026-1\S02_clases\tests\termogramas"
# img_name = "2 bhrs.JPG"
# input_path_ = os.path.join(input_folder, img_name)

output_folder = r"C:\Users\OMEN CI7\Documents\repository\Neurociencias-2026-1\S02_clases\tests\nuevos_termogramas"

legend_path = "leyenda_20_50.png"  # Imagen con la barra de colores (opcional)
max_temperature = 47.5  # <-- Ajusta aquí el máximo de TU captura (°C)
min_temperature = 20.0  # Fijo en 20 °C
cmap_vmin = 20.0  # Límite inferior del mapa de colores
cmap_vmax = 50.0  # Límite superior del mapa de colores
# -----------------------------------------------------------


# Ruta principal donde buscar imágenes
ruta_base = r"C:\Users\OMEN CI7\Documents\repository\Neurociencias-2026-1\S02_clases\tests\termogramas"  # 🔹 Cambia esto por tu ruta

# Extensiones válidas de imágenes
extensiones = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif', '.gif')

# Lista para guardar rutas encontradas
imagenes_encontradas = []

# Recorre todas las carpetas y subcarpetas
for carpeta_raiz, carpetas, archivos in os.walk(ruta_base):
    for archivo in archivos:
        if archivo.lower().endswith(extensiones):
            ruta_completa = os.path.join(carpeta_raiz, archivo)
            imagenes_encontradas.append(ruta_completa)

            subfolder = "\\".join(ruta_completa.split(sep="\\")[9:-1])
            img_name = ruta_completa.split(sep="\\")[-1]
            output_path = os.path.join(output_folder, subfolder, "procesados_" + img_name)  # PNG colorizado

            reescale_map_color(ruta_completa, output_path, min_temperature, max_temperature)

# Mostrar resultados
print("📁 Imágenes encontradas:")
for img in imagenes_encontradas:
    print(img)

print(f"\nTotal de imágenes encontradas: {len(imagenes_encontradas)}")