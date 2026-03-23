import cv2
import numpy as np 
from skimage import  io
from inferenceyolo import *
import torch
# import TorchPredict
import json
import glob
import subprocess
import pandas as pd
import time
from scipy.optimize import linear_sum_assignment
from sklearn.cluster import KMeans
import uuid
import segmentation_models_pytorch as smp
from transformers import ViTForImageClassification, ViTImageProcessor
from PIL import Image, ImageDraw
import sys
import math
import os
import shutil
from ultralytics import YOLO
from collections import Counter


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "modelos/best_model.pth"  
pretrained_model_name = "google/vit-base-patch16-224-in21k"
label_names = ['azul', 'rojo' , 'blanco'] 


model_color = ViTForImageClassification.from_pretrained(
    pretrained_model_name,
    num_labels=len(label_names),
    ignore_mismatched_sizes=True
)
model_color.classifier = torch.nn.Sequential(
    torch.nn.Dropout(0.3),
    torch.nn.Linear(model_color.config.hidden_size, len(label_names))
)
model_color.load_state_dict(torch.load(model_path, map_location=device))
model_color.to(device)
model_color.eval()


processor = ViTImageProcessor.from_pretrained(pretrained_model_name)

IMAGE_PATH = "2.jpg"

# Imágenes de análisis: todas las encontradas bajo cada subcarpeta de esta ruta.
INPUT_IMAGES_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "benchmark viajes pallet",
    "grjr57_2_101126_mediodía",
)
_BENCHMARK_IMG_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def listar_subcarpetas_inmediatas(base: str) -> list:
    """Subcarpetas directas de `base`, ordenadas por nombre."""
    if not os.path.isdir(base):
        return []
    out = []
    for nombre in sorted(os.listdir(base)):
        ruta = os.path.join(base, nombre)
        if os.path.isdir(ruta):
            out.append(ruta)
    return out


def listar_imagenes_en_carpeta(carpeta: str) -> list:
    """Rutas absolutas de imágenes bajo `carpeta` (cualquier profundidad), ordenadas."""
    if not os.path.isdir(carpeta):
        return []
    salida = []
    for root, _dirs, files in os.walk(carpeta):
        for nombre in files:
            if os.path.splitext(nombre)[1].lower() in _BENCHMARK_IMG_EXTS:
                salida.append(os.path.join(root, nombre))
    salida.sort()
    return salida




MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(device)
STD  = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(device)

# ----------------------------
# Modelo (igual que entrenamiento, pero simplificado)
# ----------------------------
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

def load_model(weights_path: str) -> torch.nn.Module:
    model_segmentacion = InferenceModel().to(device)
    state = torch.load(weights_path, map_location=device)
    model_segmentacion.load_state_dict(state, strict=True)
    model_segmentacion.eval()
    return model_segmentacion


def preprocess_image(image_path: str):
    original_bgr = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if original_bgr is None:
        raise FileNotFoundError(f"No se pudo leer la imagen: {image_path}")
    gray = cv2.cvtColor(original_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray,  (320, 320), interpolation=cv2.INTER_AREA) 
    gray_1c = np.expand_dims(gray, axis=-1)
    img_3c = np.repeat(gray_1c, 3, axis=2)  
    x = torch.from_numpy(img_3c).float().permute(2, 0, 1) / 255.0
    x = x.unsqueeze(0).to(device) 
    return x, original_bgr, gray


@torch.inference_mode()
def run_segmentation(model, image_path: str):
    x, original_bgr, resized_gray = preprocess_image(image_path)

    logits = model(x)                 # (1,1,H,W)
    prob = torch.sigmoid(logits)[0,0] # (H,W) en [0,1]
    mask = (prob > 0.5).byte().cpu().numpy()  

    return mask



def clasificar_imagen(img_bgr):
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    pil_img = Image.fromarray(img_rgb)
    inputs = processor(images=pil_img, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model_color(**inputs)
        pred = torch.argmax(outputs.logits, dim=1).item()
        probas = torch.softmax(outputs.logits, dim=1).cpu().numpy().flatten()
    return label_names[pred], probas[pred], dict(zip(label_names, probas))


def regresion_x_desde_y(boxes):
    segmentos = [
            [0,99],
            [100,199],
            [200,299],
            [300,399],
            [400,499],
            [500,607]
        ]
    xs = []
    ys = []
    
    for i , s in enumerate(segmentos):
        minx = 608
        maxx = -1
        for b in boxes:
            if( s[0] <=  b[7] <= s[1] ):
                if(b[6] < minx):
                    minx = b[6]
                if(b[6] > maxx):
                    maxx = b[6]
        if((maxx - minx) > 250):
            xs.append((maxx + minx) / 2)
            ys.append((s[0] + s[1]) / 2)
        segmentos[i].append(minx)
        segmentos[i].append(maxx)
    
    xs = np.array(xs)
    ys = np.array(ys)
    a, b = np.polyfit(ys, xs, 1)
    def f(y):
        return a * y + b
    while(True):
        if(segmentos[0][3] == -1):
            del segmentos[0]
        else:
            break
    segmentos[0][0] = 0 
    while(True):
        if(segmentos[-1][3] == -1):
            del segmentos[-1]
        else:
            break
    segmentos[-1][1] = 607 
    
    return f, a, b , segmentos


def max_y_en_x(mask, x):
    ys = np.where(mask[:, x] == 1)[0]
    if len(ys) == 0:
        return None 
    return ys.max()

def angulo_entre_puntos(x1, y1, x2, y2):    
    dx = x2 - x1
    dy = y2 - y1
    angulo_rad = math.atan2(dy, dx)
    angulo_deg = math.degrees(angulo_rad)
    return angulo_deg


def mask_to_rowwise_thirds(mask01: np.ndarray) -> np.ndarray:
    """
    mask01: matriz 2D con valores 0/1 (o bool). 1 indica píxel de máscara.
    Retorna: matriz 2D uint8 con 0 fuera de máscara y:
             1 = tercio izquierdo (por fila)
             2 = tercio medio (por fila)
             3 = tercio derecho (por fila)
    """
    m = (mask01.astype(bool))
    h, w = m.shape
    out = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        xs = np.flatnonzero(m[y])
        if xs.size == 0:
            continue

        x_min = xs[0]
        x_max = xs[-1]
        span = x_max - x_min + 1  # ancho ocupado en esa fila

        # límites de tercios (en índice absoluto de columna)
        t1_end = x_min + int(span  * 0.35) - 1
        t2_end = x_min + int(span  * 0.65) - 1

        # Etiquetar SOLO donde hay máscara=1
        # (si hay "huecos" en la fila, igual se respeta: sólo se etiqueta donde m[y,x]=1)
        for x in xs:
            if x <= t1_end:
                out[y, x] = 1
            elif x <= t2_end:
                out[y, x] = 2
            else:
                out[y, x] = 3

    return out

model_segmentacion = load_model("modelos/unet_resnet34.pth")


MODEL_PATH = "modelos/best.pt"


IMG_SIZE = 608
CONF_THRESHOLD = 0.3
IOU_THRESHOLD = 0.3
DEVICE = 0


model = YOLO(MODEL_PATH)


def _ejecutar_pipeline_lote(rutas_origen, carga_id=None):
    """Procesa un solo lote: imágenes que pertenecen a la misma subcarpeta."""
    if not rutas_origen:
        raise ValueError("rutas_origen no puede estar vacío")

    masks = {}
    unique_id = str(uuid.uuid4())
    max_coeficiente = 0
    max_archivo = ""
    max_post = ""
    validos = []
    for idx, ruta_origen in enumerate(rutas_origen):
        imagen = f"imagen{idx + 1}"
        extension = os.path.splitext(ruta_origen)[1].lower().lstrip(".") or "jpg"
        codigo = unique_id + "_" + imagen
        archivo = os.path.join("salida", f"{codigo}.{extension}")
        shutil.copy2(ruta_origen, archivo)
        tmp_img = cv2.imread(archivo)
        if tmp_img is None:
            continue
        coeficiente = 0
        mask = run_segmentation(model_segmentacion, archivo)
        mask = clean_and_dilate(mask)
        masks[imagen] = mask
        mask2 = mask.copy()
        mask2[mask2 > 0] = 255
        cv2.imwrite("salida/" + codigo + "_mask." + extension, mask2)
        coeficiente_base = 0
        dx = -1
        maxx = 0
        for x in range(320):
            m = max_y_en_x(mask, x)
            if m:
                if m > maxx:
                    maxx = m
                    dx = x
        mld = dx
        ux = None
        for x in range(dx, 320, 10):
            if ux is not None:
                try:
                    y1 = max_y_en_x(mask, ux)
                    y2 = max_y_en_x(mask, x)
                    if (y1 - y2) < 10:
                        mld = x
                except Exception:
                    pass
            ux = x
        mli = dx
        ux = None
        for x in range(dx, 0, -10):
            if ux is not None:
                try:
                    y1 = max_y_en_x(mask, ux)
                    y2 = max_y_en_x(mask, x)
                    if (y1 - y2) < 10:
                        mli = x
                except Exception:
                    pass
            ux = x
        dli = dx - mli
        dld = mld - dx
        if dli > dld:
            if dld == 0:
                coeficiente_base = 1
            else:
                coeficiente_base = 1
                coeficiente_base = 1 if coeficiente_base > 1 else coeficiente_base
            y1 = max_y_en_x(mask, mli)
            y2 = max_y_en_x(mask, dx)
            angulo = abs(angulo_entre_puntos(mli, y1, dx, y2))
        else:
            if dli == 0:
                coeficiente_base = 1
            else:
                coeficiente_base = 1
                coeficiente_base = 1 if coeficiente_base > 1 else coeficiente_base
            y1 = max_y_en_x(mask, dx)
            y2 = max_y_en_x(mask, mld)
            angulo = abs(angulo_entre_puntos(dx, y1, mld, y2))

        coeficiente_base = coeficiente_base - (angulo * 0.01)
        coeficiente_base = 0 if coeficiente_base < 0 else coeficiente_base

        coeficiente_izquierdo = 1 - mask[:, 0:10].mean()
        coeficiente_derecho = 1 - mask[:, 310:].mean()
        coeficiente_inferior = 1 - mask[310:, :].mean()
        coeficiente = (
            (coeficiente_base * 1)
            + coeficiente_izquierdo
            + coeficiente_derecho
            + coeficiente_inferior
        ) / 4
        if (tmp_img.shape[0] / tmp_img.shape[1]) > 2:
            coeficiente = 0
        print(imagen, coeficiente)
        if coeficiente > max_coeficiente:
            max_coeficiente = coeficiente
            max_archivo = archivo
            max_post = imagen
        if coeficiente > 0.91:
            validos.append(imagen)

    if not max_post:
        raise RuntimeError(
            "No se pudo leer ninguna imagen válida en las rutas de este lote."
        )

    extension = os.path.splitext(max_archivo)[1].lower().lstrip(".") or "jpg"

    if (len(validos) < 3) or True:
        cuenta_colores = {"azul": 0, "rojo": 0, "blanco": 0}
        mask = masks[max_post]
        mask_dilatado = clean_and_dilate(mask, 30)
        sectores = mask_to_rowwise_thirds(mask_dilatado)
        mask_dilatado = clean_and_dilate(mask, 5)
        sectores[mask_dilatado == 0] = 0
        sectores = cv2.resize(
            sectores.astype(np.uint8),
            (608, 608),
            interpolation=cv2.INTER_NEAREST,
        )
        frame_original = cv2.imread(max_archivo)
        frame = cv2.resize(frame_original, (608, 608))
        results = model.predict(
            source=frame,
            imgsz=IMG_SIZE,
            conf=CONF_THRESHOLD,
            iou=IOU_THRESHOLD,
            device=DEVICE,
            augment=False,
            verbose=False,
        )
        r = results[0]
        boxes = []
        for i, box in enumerate(r.boxes):
            cls_id = int(box.cls.item())
            conf = float(box.conf.item())
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            xyboxes = [int(x1), int(y1), int(x2), int(y2), 0, -1]
            cx = centroide(xyboxes)
            xyboxes.extend(cx)
            clase_predicha, _, _ = clasificar_imagen(
                frame[int(y1) : int(y2), int(x1) : int(x2)]
            )
            xyboxes.append(clase_predicha)
            boxes.append(xyboxes)

        groups = {0: [], 1: [], 2: []}
        for box in boxes:
            x = sectores[box[7], box[6]]
            if x != 0:
                groups[x - 1].append(box)
        for k in groups:
            groups[k] = sorted(groups[k], key=lambda x: x[7], reverse=False)

        if len(groups[0]) == len(groups[1]) == len(groups[2]):
            imgau = graficar(frame, groups[0])
            iimgau = Image.fromarray(imgau)
            draw = ImageDraw.Draw(iimgau)
            orden_por_clase = {"azul": 0, "rojo": 0, "blanco": 0}
            for i, g in enumerate(groups[0]):
                colores = [
                    groups[0][i][-1],
                    groups[1][i][-1],
                    groups[2][i][-1],
                ]
                clase_predicha = Counter(colores).most_common(1)[0][0]
                orden_por_clase[clase_predicha] += 1
                etiqueta = f"{clase_predicha} ({orden_por_clase[clase_predicha]})"
                draw.text((int(g[2]), int(g[1])), etiqueta, fill=(0, 255, 0))
                cuenta_colores[clase_predicha] += 1
        else:
            i_cuenta = -1
            cuenta = 0
            for k, v in groups.items():
                if len(v) > cuenta:
                    i_cuenta = k
                    cuenta = len(v)
            nombre_columna = {0: "left", 1: "center", 2: "right"}.get(i_cuenta, "unknown")
            print(f"Columna seleccionada: {nombre_columna} (grupo {i_cuenta}, {cuenta} detecciones)")
            imgau = graficar(frame, groups[i_cuenta])
            iimgau = Image.fromarray(imgau)
            draw = ImageDraw.Draw(iimgau)
            orden_por_clase = {"azul": 0, "rojo": 0, "blanco": 0}
            for g in groups[i_cuenta]:
                clase_predicha = g[-1]
                orden_por_clase[clase_predicha] += 1
                etiqueta = f"{clase_predicha} ({orden_por_clase[clase_predicha]})"
                draw.text((int(g[2]), int(g[1])), etiqueta, fill=(0, 255, 0))
                cuenta_colores[clase_predicha] += 1

    else:
        cuenta_colores = {"azul": 0, "rojo": 0, "blanco": 0}
        l_cuenta_colores = {"azul": 0, "rojo": 0, "blanco": 0}
        for post_img in validos:
            arch_loop = "salida/" + unique_id + "_" + post_img + "." + extension
            mask = masks[post_img]
            mask_dilatado = clean_and_dilate(mask, 30)
            sectores = mask_to_rowwise_thirds(mask_dilatado)
            mask_dilatado = clean_and_dilate(mask, 5)
            sectores[mask_dilatado == 0] = 0
            sectores = cv2.resize(
                sectores.astype(np.uint8),
                (608, 608),
                interpolation=cv2.INTER_NEAREST,
            )
            frame_original = cv2.imread(arch_loop)
            frame = cv2.resize(frame_original, (608, 608))
            results = model.predict(
                source=frame,
                imgsz=IMG_SIZE,
                conf=CONF_THRESHOLD,
                iou=IOU_THRESHOLD,
                device=DEVICE,
                augment=False,
                verbose=False,
            )
            r = results[0]
            boxes = []
            for i, box in enumerate(r.boxes):
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                x1, y1, x2, y2 = box.xyxy[0].tolist()
                xyboxes = [int(x1), int(y1), int(x2), int(y2), 0, -1]
                cx = centroide(xyboxes)
                xyboxes.extend(cx)
                clase_predicha, _, _ = clasificar_imagen(
                    frame[int(y1) : int(y2), int(x1) : int(x2)]
                )
                xyboxes.append(clase_predicha)
                boxes.append(xyboxes)

            groups = {0: [], 1: [], 2: []}
            for box in boxes:
                x = sectores[box[7], box[6]]
                if x != 0:
                    groups[x - 1].append(box)
            for k in groups:
                groups[k] = sorted(groups[k], key=lambda x: x[7], reverse=False)

            if len(groups[0]) == len(groups[1]) == len(groups[2]):
                imgau = graficar(frame, groups[0])
                iimgau = Image.fromarray(imgau)
                draw = ImageDraw.Draw(iimgau)
                orden_por_clase = {"azul": 0, "rojo": 0, "blanco": 0}
                for i, g in enumerate(groups[0]):
                    colores = [
                        groups[0][i][-1],
                        groups[1][i][-1],
                        groups[2][i][-1],
                    ]
                    clase_predicha = Counter(colores).most_common(1)[0][0]
                    orden_por_clase[clase_predicha] += 1
                    etiqueta = f"{clase_predicha} ({orden_por_clase[clase_predicha]})"
                    draw.text((int(g[2]), int(g[1])), etiqueta, fill=(0, 255, 0))
                    l_cuenta_colores[clase_predicha] += 1
            else:
                i_cuenta = -1
                cuenta = 0
                for k, v in groups.items():
                    if len(v) > cuenta:
                        i_cuenta = k
                        cuenta = len(v)
                imgau = graficar(frame, groups[i_cuenta])
                iimgau = Image.fromarray(imgau)
                draw = ImageDraw.Draw(iimgau)
                orden_por_clase = {"azul": 0, "rojo": 0, "blanco": 0}
                for g in groups[i_cuenta]:
                    clase_predicha = g[-1]
                    orden_por_clase[clase_predicha] += 1
                    etiqueta = f"{clase_predicha} ({orden_por_clase[clase_predicha]})"
                    draw.text((int(g[2]), int(g[1])), etiqueta, fill=(0, 255, 0))
                    l_cuenta_colores[clase_predicha] += 1
        print("???", l_cuenta_colores)
        n_validos = max(len(validos), 1)
        cuenta_colores["azul"] = round(l_cuenta_colores["azul"] / n_validos)
        cuenta_colores["blanco"] = round(l_cuenta_colores["blanco"] / n_validos)
        cuenta_colores["rojo"] = round(l_cuenta_colores["rojo"] / n_validos)

    imgau = np.array(iimgau)
    io.imshow(imgau[:, :, ::-1])
    archivo_resultado = "salida/" + unique_id + "_" + max_post + "_resultado." + extension
    cv2.imwrite(archivo_resultado, imgau)
    abs_resultado = os.path.abspath(archivo_resultado)
    cuenta_colores["imagen"] = max_post
    cuenta_colores["ID"] = unique_id
    cuenta_colores["ruta_resultado"] = abs_resultado
    uid_prefix = unique_id + "_"
    for arch in glob.glob(os.path.join("salida", "*")):
        if os.path.abspath(arch) == abs_resultado:
            continue
        if os.path.basename(arch).startswith(uid_prefix):
            os.remove(arch)
    print("!!!", carga_id, cuenta_colores)
    return cuenta_colores


def ejecutar_analisis(
    area="",
    carga_id=None,
    centro="",
    patente="",
    fecha=None,
):
    """
    Ejecuta el pipeline por cada **subcarpeta directa** de INPUT_IMAGES_DIR
    (benchmark viajes pallet / grjr57_2_101126_mediodía). Las imágenes de distintas
    subcarpetas no se mezclan. Devuelve una lista de resultados (uno por subcarpeta
    con al menos una imagen).
    Metadatos opcionales (también vía env: SPOT_AREA, SPOT_CARGA_ID, SPOT_CENTRO, SPOT_PATENTE, SPOT_FECHA).
    """
    if not fecha:
        fecha = time.strftime("%Y-%m-%d %H:%M:%S")

    os.makedirs("salida", exist_ok=True)
    if not os.path.isdir(INPUT_IMAGES_DIR):
        raise FileNotFoundError(f"No existe la carpeta: {INPUT_IMAGES_DIR}")

    resultados = []
    for carpeta in listar_subcarpetas_inmediatas(INPUT_IMAGES_DIR):
        rutas = listar_imagenes_en_carpeta(carpeta)
        if not rutas:
            continue
        print(f"Procesando subcarpeta: {os.path.basename(carpeta)}")
        out = _ejecutar_pipeline_lote(rutas, carga_id=carga_id)
        out["subcarpeta"] = os.path.basename(carpeta)
        out["ruta_subcarpeta"] = os.path.abspath(carpeta)
        resultados.append(out)

    if not resultados:
        raise FileNotFoundError(
            f"No hay imágenes en ninguna subcarpeta directa de: {INPUT_IMAGES_DIR}"
        )
    return resultados


if __name__ == "__main__":
    try:
        out = ejecutar_analisis(
            area=os.environ.get("SPOT_AREA", ""),
            carga_id=os.environ.get("SPOT_CARGA_ID"),
            centro=os.environ.get("SPOT_CENTRO", ""),
            patente=os.environ.get("SPOT_PATENTE", ""),
            fecha=os.environ.get("SPOT_FECHA"),
        )
        print(json.dumps(out, ensure_ascii=False, indent=2))
    except Exception as exc:
        print(
            json.dumps({"estado": "error", "descripcion": str(exc)}, ensure_ascii=False),
            file=sys.stderr,
        )
        sys.exit(1)
