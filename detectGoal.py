max_frames = 500  # ajustá este número según lo que necesites
start_frame = 200 # Define el número de frame donde quieres empezar
filename='/content/video3.webm.mkv'

import torch
from ultralytics import YOLO
import cv2
from google.colab.patches import cv2_imshow
from google.colab import userdata
from inference import get_model

# Seleccionar el mejor dispositivo disponible
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Usando dispositivo: {device}")

# Cargar modelo YOLOv8 (puede ser 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt' o 'yolov8x.pt')
model = YOLO('yolov8s.pt')  # Puedes cambiar por 'yolov8x.pt' si tienes buena GPU
model.to(device)



# Cargar el modelo entrenado
modelBall = get_model(model_id="yolov8-ball-and-player-detection/1", api_key=userdata.get('ROBOFLOW'))



import cv2
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

def extract_two_colors_no_grass(frame, box, k=2):
    x1, y1, x2, y2 = map(int, box)
    h, w = y2 - y1, x2 - x1
    if h < 10 or w < 10:
        return None

    # ROI ampliado torso
    x_start = x1 + int(0.15 * w)
    x_end = x2 - int(0.15 * w)
    y_start = y1 + int(0.1 * h)
    y_end = y1 + int(0.6 * h)
    roi = frame[y_start:y_end, x_start:x_end]

    if roi.size == 0:
        return None

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Máscara para excluir césped (verde)
    lower_green = np.array([35, 40, 40])
    upper_green = np.array([85, 255, 255])
    mask_grass = cv2.inRange(hsv, lower_green, upper_green)
    mask_no_grass = cv2.bitwise_not(mask_grass)

    # Filtrar sólo píxeles sin césped
    pixels = roi[mask_no_grass > 0]
    if len(pixels) < 20:
        pixels = roi.reshape(-1, 3)
    if len(pixels) == 0:
        return None

    kmeans = KMeans(n_clusters=k, n_init='auto')
    kmeans.fit(pixels)
    counts = np.bincount(kmeans.labels_)
    dominant_indices = np.argsort(counts)[::-1]
    dominant_colors = kmeans.cluster_centers_[dominant_indices].astype(int)

    return dominant_colors

def filter_partial_players(boxes, frame_shape, margin=10):
    frame_h, frame_w = frame_shape[:2]
    filtered = []
    for det in boxes:
        if int(det.cls) != 0:
            continue
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        if x1 <= margin or y1 <= margin or x2 >= frame_w - margin or y2 >= frame_h - margin:
            continue
        filtered.append(det)
    return filtered

previous_team_avg_colors = []
def color_distance(c1, c2):
    # distancia euclidiana entre dos colores BGR
    return np.linalg.norm(np.array(c1) - np.array(c2))

def assign_team_by_color(player_colors, previous_avg_colors, max_dist=80):
    """
    Asigna equipo a cada jugador comparando su primer color dominante con los colores promedio previos.
    Si no hay color parecido, se asigna un equipo nuevo hasta 4 máximo.
    player_colors: lista de np.array con 2 colores dominantes (BGR)
    previous_avg_colors: lista de colores promedio (BGR) usados en el frame anterior
    """
    labels = []
    new_avg_colors = previous_avg_colors.copy()

    for colors in player_colors:
        c = colors[0]  # primer color dominante para comparación
        if len(new_avg_colors) == 0:
            # primer frame: asignar equipo 0 y agregar color promedio
            labels.append(0)
            new_avg_colors.append(c)
            continue

        # Buscar color más cercano en new_avg_colors
        dists = [color_distance(c, prev_c) for prev_c in new_avg_colors]
        min_dist = min(dists)
        min_idx = dists.index(min_dist)

        if min_dist < max_dist:
            labels.append(min_idx)
            # actualizar color promedio suavemente
            new_avg_colors[min_idx] = 0.7 * np.array(new_avg_colors[min_idx]) + 0.3 * c
        else:
            # Si no hay cercano y no superamos 4 equipos, agregamos uno nuevo
            if len(new_avg_colors) < 4:
                labels.append(len(new_avg_colors))
                new_avg_colors.append(c)
            else:
                # Asignar equipo más cercano aunque esté lejos (fallback)
                labels.append(min_idx)
                new_avg_colors[min_idx] = 0.7 * np.array(new_avg_colors[min_idx]) + 0.3 * c

    return labels, new_avg_colors

def draw_players(frame, boxes, team_labels, team_colors_map):
    for det, label in zip(boxes, team_labels):
        x1, y1, x2, y2 = map(int, det.xyxy[0].tolist())
        color = team_colors_map[label]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
    return frame


import numpy as np
import cv2

def compute_goal_homography(top_left, top_right, base_right, base_left):
    """
    Calcula la homografía que transforma el área del arco a un plano frontal.

    Parámetros:
    - top_left, top_right, base_right, base_left: puntos (x, y) en imagen

    Retorna:
    - H: matriz de homografía 3x3
    """

    src_pts = np.array([top_left, top_right, base_right, base_left], dtype=np.float32)

    # Definir plano destino (por ejemplo, 2 m ancho x 1 m alto)
    dst_pts = np.array([
        [0, 0],
        [2.0, 0],
        [2.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)

    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

def check_goal(ball_center, H):
    """
    ball_center: (x, y) en píxeles
    H: homografía del arco
    """
    if H is None or ball_center is None:
        return False

    point = np.array([[ball_center]], dtype=np.float32)
    field_pos = cv2.perspectiveTransform(point, H)[0][0]  # (x, y) en espacio del arco

    x, y = field_pos
    return (0 <= x <= 2.0) and (0 <= y <= 1.0)

# Colores para dibujar cada equipo (4 colores bien distintos)
team_colors_map = {
    0: (255, 0, 0),    # azul
    1: (0, 255, 0),    # verde
    2: (0, 0, 255),    # rojo
    3: (0, 255, 255),  # amarillo
}
is_goal=False

def process_frame(frame):
    global previous_team_avg_colors
    global is_goal

    results = model(frame)[0]
    filtered_boxes = filter_partial_players(results.boxes, frame.shape)
    player_colors = []
    valid_boxes = []

    for det in filtered_boxes:
        box = det.xyxy[0].tolist()
        colors = extract_two_colors_no_grass(frame, box)
        if colors is not None:
            player_colors.append(colors)
            valid_boxes.append(det)

    team_labels, previous_team_avg_colors = assign_team_by_color(player_colors, previous_team_avg_colors)

    frame_out, center_ball = draw_ball(frame.copy())
    frame_out = draw_players(frame_out, valid_boxes, team_labels, team_colors_map)
    frame_out, H, goal_corners = detect_goalpost(frame_out)
    if not is_goal:
      is_goal = check_goal(center_ball, H)
    if is_goal:
      print("¡GOL!")
      cv2.putText(frame_out, "GOL", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 4)

    return frame_out

import supervision as sv

def callback(patch: np.ndarray) -> sv.Detections:
    result = modelBall.infer(patch, confidence=0.1)[0]
    return sv.Detections.from_inference(result)





# Inicializar el rastreador una sola vez
tracker = sv.ByteTrack()  # o el tracker que prefieras

# Opcional: para mostrar trayectorias o anotaciones visuales
annotator = sv.TriangleAnnotator(color=sv.Color.from_hex('#FF1493'), height=20, base=25)

def draw_ball(frame):
    global tracker

    h, w, _ = frame.shape
    # Paso 1: detección con el modelo
    slicer = sv.InferenceSlicer(
      callback = callback,
      overlap_filter = sv.OverlapFilter.NON_MAX_SUPPRESSION,
      slice_wh = (w // 2 + 100, h // 2 + 100),
      overlap_ratio_wh = None,
      overlap_wh = (100, 100),
      iou_threshold = 0.1
    )

    detections = slicer(frame)

    # Paso 2: actualizar tracker
    tracked = tracker.update_with_detections(detections)

    # Paso 3: visualizar pelota(s) detectada(s)
    if tracked:
        # Anotar visualmente la trayectoria
        frame = annotator.annotate(scene=frame, detections=tracked)

        # Podemos devolver el centro de la primera pelota detectada (opcional)
        x1, y1, x2, y2 = tracked.xyxy[0]
        cx = int((x1 + x2) / 2)
        cy = int((y1 + y2) / 2)
        center_ball = (cx, cy)
    else:
        center_ball = None

    return frame, center_ball

import numpy as np
import cv2

def line_intersection(line1, line2):
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2)*(y3 - y4) - (y1 - y2)*(x3 - x4)
    if denom == 0:
        return None

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denom

    if (min(x1,x2)-5 <= px <= max(x1,x2)+5 and min(y1,y2)-5 <= py <= max(y1,y2)+5 and
        min(x3,x4)-5 <= px <= max(x3,x4)+5 and min(y3,y4)-5 <= py <= max(y3,y4)+5):
        return int(px), int(py)
    return None

def compute_goal_homography(top_left, top_right, base_right, base_left):
    src_pts = np.array([top_left, top_right, base_right, base_left], dtype=np.float32)
    dst_pts = np.array([
        [0, 0],
        [2.0, 0],
        [2.0, 1.0],
        [0.0, 1.0]
    ], dtype=np.float32)
    H, _ = cv2.findHomography(src_pts, dst_pts)
    return H

last_corners = None
last_homography = None

def detect_goalpost(frame):
    global last_homography, last_corners

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_white = np.array([0, 0, 200])
    upper_white = np.array([180, 50, 255])
    mask_white = cv2.inRange(hsv, lower_white, upper_white)

    lines = cv2.HoughLinesP(mask_white, 1, np.pi / 360, threshold=100, minLineLength=80, maxLineGap=20)
    if lines is None:
        return frame, last_homography, last_corners

    verticals, horizontals = [], []

    for line in lines:
        x1, y1, x2, y2 = line[0]
        dx, dy = abs(x2 - x1), abs(y2 - y1)
        if dx < 10 and dy > 50:
            verticals.append((x1, y1, x2, y2))
        elif dy < 50 and dx > 100:
            horizontals.append((x1, y1, x2, y2))

    best_triplet = None
    best_score = 0

    for i in range(len(verticals)):
        for j in range(i+1, len(verticals)):
            v1, v2 = verticals[i], verticals[j]
            mid_v1 = (v1[0] + v1[2]) / 2
            mid_v2 = (v2[0] + v2[2]) / 2
            if abs(mid_v2 - mid_v1) < 100:
                continue
            for h in horizontals:
                inter1 = line_intersection(v1, h)
                inter2 = line_intersection(v2, h)
                if inter1 and inter2:
                    if inter1[1] < max(v1[1], v1[3]) and inter2[1] < max(v2[1], v2[3]):
                        length_v1 = abs(v1[3] - v1[1])
                        length_v2 = abs(v2[3] - v2[1])
                        length_h = abs(h[2] - h[0])
                        score = length_v1 + length_v2 + length_h
                        if score > best_score:
                            best_score = score
                            best_triplet = (v1, v2, h)

    if best_triplet:
        v1, v2, h = best_triplet
        top_left = line_intersection(v1, h)
        top_right = line_intersection(v2, h)
        base_left = (v1[0], max(v1[1], v1[3]))
        base_right = (v2[0], max(v2[1], v2[3]))

        if top_left and top_right:
            # Dibujar y calcular homografía
            pts = np.array([top_left, top_right, base_right, base_left], np.int32)
            overlay = frame.copy()
            cv2.fillPoly(overlay, [pts], (200, 100, 128))
            cv2.drawContours(frame, [pts], -1, (0, 255, 255), 2)
            frame = cv2.addWeighted(overlay, 0.3, frame, 0.7, 0)

            corners = (top_left, top_right, base_right, base_left)
            H = compute_goal_homography(*corners)

            # Guardar detección válida
            last_homography = H
            last_corners = corners

            return frame, H, corners

    # Si no hubo buena detección, usar la última válida
    return frame, last_homography, last_corners



# Abrir el video original
cap = cv2.VideoCapture()
cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
frame_count = start_frame

# Obtener dimensiones del frame
ret, frame = cap.read(filename)
if not ret:
    print("Error al leer el primer frame")
    cap.release()
else:
    height, width = frame.shape[:2]

    # Inicializar el writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # o 'XVID', 'avc1', etc.
    out = cv2.VideoWriter('/content/output3.mp4', fourcc, cap.get(cv2.CAP_PROP_FPS), (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)  # Reiniciar al frame inicial

    while True:
        ret, frame = cap.read()
        if not ret or frame_count >= max_frames:
            break

        print(f"Procesando frame {frame_count}")
        frame_out = process_frame(frame)

        out.write(frame_out)  # Escribir frame en el video

        frame_count += 1

    cap.release()
    out.release()
    print("Video guardado en /content/output3.mp4")
