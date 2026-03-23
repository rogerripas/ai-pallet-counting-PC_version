"""
pallet_heuristics.py
--------------------
Para cada imagen en 'inputs_heuristics/':
  1. Segmenta el pallet con U-Net  →  máscara binaria
  2. Detecta unidades con YOLO
  3. Descarta bounding boxes cuyo centroide quede fuera de la máscara
  4. Asigna cada bbox a columna izquierda / centro / derecha
  5. Dibuja los bboxes coloreados por columna:
       izquierda → rojo  |  centro → verde  |  derecha → azul
  6. Guarda la imagen anotada en 'outputs_heuristics/'
"""

import cv2
import numpy as np
import torch
import os
import shutil
import segmentation_models_pytorch as smp
from PIL import Image, ImageDraw
from ultralytics import YOLO
from inferenceyolo import clean_and_dilate, centroide

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR     = os.path.dirname(os.path.abspath(__file__))
INPUT_DIR    = os.path.join(BASE_DIR, "inputs_heuristics")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs_heuristics")
UNET_WEIGHTS = os.path.join(BASE_DIR, "modelos", "unet_resnet34.pth")
YOLO_WEIGHTS = os.path.join(BASE_DIR, "modelos", "best.pt")

IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}

# ---------------------------------------------------------------------------
# YOLO params
# ---------------------------------------------------------------------------
IMG_SIZE       = 608
CONF_THRESHOLD = 0.3
IOU_THRESHOLD  = 0.3
YOLO_DEVICE    = 0

# Colores BGR por columna
COLOR_LEFT   = (0,   0,   255)   # rojo
COLOR_CENTER = (0,   255, 0  )   # verde
COLOR_RIGHT  = (255, 0,   0  )   # azul
COLOR_OUT    = (80,  80,  80 )   # gris (descartados)

# ---------------------------------------------------------------------------
# U-Net
# ---------------------------------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)


class InferenceModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = smp.create_model(
            "Unet",
            encoder_name="resnet34",
            in_channels=3,
            classes=1,
        )

    def forward(self, x):
        x = (x - MEAN) / STD
        return self.model(x)


def load_unet(weights_path: str) -> torch.nn.Module:
    m = InferenceModel().to(device)
    state = torch.load(weights_path, map_location=device)
    m.load_state_dict(state, strict=True)
    m.eval()
    return m


def preprocess_for_unet(image_path: str):
    bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {image_path}")
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (320, 320), interpolation=cv2.INTER_AREA)
    gray_3c = np.repeat(np.expand_dims(gray, axis=-1), 3, axis=2)
    x = torch.from_numpy(gray_3c).float().permute(2, 0, 1) / 255.0
    x = x.unsqueeze(0).to(device)
    return x, bgr


@torch.inference_mode()
def run_segmentation(unet, image_path: str) -> np.ndarray:
    """Devuelve máscara binaria 0/1 de tamaño 320×320."""
    x, _ = preprocess_for_unet(image_path)
    logits = unet(x)
    prob   = torch.sigmoid(logits)[0, 0]
    mask   = (prob > 0.5).byte().cpu().numpy()
    return mask


# ---------------------------------------------------------------------------
# Separación en tercios por fila (igual que index.py)
# ---------------------------------------------------------------------------

def mask_to_rowwise_thirds(mask01: np.ndarray) -> np.ndarray:
    """
    Retorna matriz 2D uint8 con:
      0 = fuera de máscara
      1 = tercio izquierdo
      2 = tercio central
      3 = tercio derecho
    """
    m   = mask01.astype(bool)
    h, w = m.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        xs = np.flatnonzero(m[y])
        if xs.size == 0:
            continue
        x_min, x_max = xs[0], xs[-1]
        span  = x_max - x_min + 1
        t1_end = x_min + int(span * 0.35) - 1
        t2_end = x_min + int(span * 0.65) - 1
        for x in xs:
            if x <= t1_end:
                out[y, x] = 1
            elif x <= t2_end:
                out[y, x] = 2
            else:
                out[y, x] = 3
    return out


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def listar_imagenes(carpeta: str) -> list:
    rutas = []
    for nombre in sorted(os.listdir(carpeta)):
        if os.path.splitext(nombre)[1].lower() in IMG_EXTS:
            rutas.append(os.path.join(carpeta, nombre))
    return rutas


def dibujar_boxes(frame_bgr: np.ndarray, groups: dict, boxes_fuera: list, mask_608: np.ndarray) -> np.ndarray:
    """
    groups: {0: [boxes izquierda], 1: [boxes centro], 2: [boxes derecha]}
    Colores: izquierda=rojo, centro=verde, derecha=azul, fuera=gris
    """
    img  = Image.fromarray(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img)

    col_rgb = {
        0: (255, 0,   0  ),   # izquierda → rojo
        1: (0,   255, 0  ),   # centro    → verde
        2: (0,   0,   255),   # derecha   → azul
    }
    col_labels = {0: "L", 1: "C", 2: "R"}

    # Descartados (fuera de máscara)
    for box in boxes_fuera:
        draw.rectangle([box[0], box[1], box[2], box[3]], outline=(128, 128, 128), width=2)

    # Por columna
    for col_idx, boxes in groups.items():
        color = col_rgb[col_idx]
        for box in boxes:
            draw.rectangle([box[0], box[1], box[2], box[3]], outline=color, width=2)
            cx, cy = box[6], box[7]
            draw.ellipse([cx - 3, cy - 3, cx + 3, cy + 3], fill=color)
            draw.text((box[0] + 2, box[1] + 2), col_labels[col_idx], fill=color)

    result_bgr = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

    # Contorno de la máscara en cian
    contornos, _ = cv2.findContours(mask_608, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result_bgr, contornos, -1, (255, 200, 0), 2)

    return result_bgr


# ---------------------------------------------------------------------------
# Pipeline por imagen
# ---------------------------------------------------------------------------

def procesar_imagen(ruta: str, unet, yolo_model) -> dict:
    # Diccionarios
    dict_planks = {}
    dict_pallets = {}
    
    nombre = os.path.basename(ruta)
    print(f"\n{'='*60}")
    print(f"  Procesando: {nombre}")
    print(f"{'='*60}")

    # 1. Segmentación U-Net → máscara 320×320
    mask_raw = run_segmentation(unet, ruta)
    mask     = clean_and_dilate(mask_raw, 5)

    # Sectores con dilatación mayor (igual que index.py línea 382-385)
    mask_dilatado = clean_and_dilate(mask_raw, 30)
    sectores = mask_to_rowwise_thirds(mask_dilatado)
    mask_dilatado_5 = clean_and_dilate(mask_raw, 5)
    sectores[mask_dilatado_5 == 0] = 0

    # Escalar máscara y sectores a 608×608
    mask_608 = cv2.resize(
        mask.astype(np.uint8),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )
    sectores_608 = cv2.resize(
        sectores.astype(np.uint8),
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_NEAREST,
    )

    # 2. Frame para YOLO
    frame_original = cv2.imread(ruta)
    frame = cv2.resize(frame_original, (IMG_SIZE, IMG_SIZE))

    # 3. Detección YOLO
    results = yolo_model.predict(
        source=frame,
        imgsz=IMG_SIZE,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        device=YOLO_DEVICE,
        augment=False,
        verbose=False,
    )
    r = results[0]

    boxes_fuera = []
    groups = {0: [], 1: [], 2: []}   # 0=izq, 1=centro, 2=der

    for i, box in enumerate(r.boxes):
        x1, y1, x2, y2 = box.xyxy[0].tolist()
        xybox = [int(x1), int(y1), int(x2), int(y2), 0, -1]
        xybox.extend(centroide(xybox))   # [6]=cx, [7]=cy

        cx, cy = xybox[6], xybox[7]
        sector = sectores_608[cy, cx]    # 0=fuera, 1=izq, 2=centro, 3=der

        if sector == 0:
            boxes_fuera.append(xybox)
        else:
            groups[sector - 1].append(xybox)
            
            # Agregar tablones al diccionario 
            plank_id = f"plank_{i}"
            dict_planks[plank_id] = {
                "id": i,
                "bbox": [xybox[0], xybox[1], xybox[2], xybox[3]],
                "center": [xybox[6], xybox[7]],
                "column": int(sector - 1), # 0=izq, 1=centro, 2=der
            }

        # print(f"  xyboxes: {xybox}  →  sector={sector}")

    # Ordenar cada grupo por Y (de arriba a abajo)
    for k in groups:
        groups[k] = sorted(groups[k], key=lambda b: b[7])

    # total_dentro = sum(len(v) for v in groups.values())
    # print(f"\n  Detecciones totales : {total_dentro + len(boxes_fuera)}")
    # print(f"  Izquierda (rojo)    : {len(groups[0])}")
    # print(f"  Centro    (verde)   : {len(groups[1])}")
    # print(f"  Derecha   (azul)    : {len(groups[2])}")
    # print(f"  Descartadas (gris)  : {len(boxes_fuera)}")

    # # 4. Imagen anotada
    # frame_anotado = dibujar_boxes(frame, groups, boxes_fuera, mask_608)
    
    # return {
    #     "nombre": nombre,
    #     "groups": groups,
    #     "boxes_fuera": boxes_fuera,
    #     "frame_anotado": frame_anotado,
    # }
    
    return dict_planks, groups


# ---------------------------------------------------------------------------
# Filtros
# ---------------------------------------------------------------------------
def filtro_1(ruta: str, groups: dict) -> bool:
    """
    Mueve la imagen original a 'filtros/filtro_1'
    si alguna de las columnas (izquierda, centro, derecha) no tiene detecciones.
    Retorna True si la imagen fue filtrada, False si pasó el filtro.
    """
    if any(len(v) == 0 for v in groups.values()):
        destino_dir = os.path.join(BASE_DIR, "filtros", "filtro_1")
        os.makedirs(destino_dir, exist_ok=True)
        destino = os.path.join(destino_dir, os.path.basename(ruta))
        shutil.move(ruta, destino)
        columnas_vacias = [k for k, v in groups.items() if len(v) == 0]
        nombres = {0: "izquierda", 1: "centro", 2: "derecha"}
        vacias_str = ", ".join(nombres[k] for k in columnas_vacias)
        print(f"  [FILTRO 1] Columna(s) vacía(s): {vacias_str}  →  movida a: {destino}")
        return True
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def ejecutar():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"No existe la carpeta de entrada: {INPUT_DIR}")

    rutas = listar_imagenes(INPUT_DIR)
    if not rutas:
        raise FileNotFoundError(f"No hay imágenes en: {INPUT_DIR}")

    print(f"Cargando U-Net desde  : {UNET_WEIGHTS}")
    unet = load_unet(UNET_WEIGHTS)

    print(f"Cargando YOLO desde   : {YOLO_WEIGHTS}")
    yolo_model = YOLO(YOLO_WEIGHTS)

    print(f"\nImágenes encontradas  : {len(rutas)}")

    for ruta in rutas:
        # resultado = procesar_imagen(ruta, unet, yolo_model)
        # ruta_salida = os.path.join(OUTPUT_DIR, resultado["nombre"])
        # cv2.imwrite(ruta_salida, resultado["frame_anotado"])
        # print(f"\n  Guardado en: {ruta_salida}")
        
        dict_planks, groups = procesar_imagen(ruta, unet, yolo_model)
        filtro_1(ruta, groups)
        # print(dict_planks)
        
    print(f"\n{'='*60}")
    print(f"Listo. Resultados en: {OUTPUT_DIR}")
    

if __name__ == "__main__":
    ejecutar()
