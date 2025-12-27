import cv2
import torch
import os
import numpy as np
from playsound import playsound
from datetime import datetime
from collections import defaultdict, deque
import threading
from typing import Optional

#------------------НАСТРОЙКИ------------------
CONF_THRESHOLD: float = 0.5
MIN_OBJECT_PERCENT: float = 0.02
MAX_TRAJECTORY_LEN: int = 30

IMPORTANT_OBJECTS: list[str] = ['person', 'car']
SOUND_PATH: str = "sounds/840060__pedr01__notification.wav"
SOUND_COOLDOWN: int = 3  #секунди між звуками

#створює папку для скріншотів, якщо її немає
if not os.path.exists('screenshots'):
    os.makedirs('screenshots')

#------------------ПЕРЕКЛАД КЛАСІВ------------------
YOLO_CLASSES_UA: dict[str, str] = {
    'person': 'людина',
    'bicycle': 'велосипед',
    'car': 'автомобіль',
    'motorcycle': 'мотоцикл',
    'airplane': 'літак',
    'bus': 'автобус',
    'train': 'потяг',
    'truck': 'вантажівка',
    'boat': 'човен',
    'traffic light': 'світлофор',
    'fire hydrant': 'пожежний гідрант',
    'stop sign': 'знак стоп',
    'parking meter': 'паркомат',
    'bench': 'лавка',
    'bird': 'птах',
    'cat': 'кіт',
    'dog': 'собака',
    'horse': 'кінь',
    'sheep': 'вівця',
    'cow': 'корова',
    'elephant': 'слон',
    'bear': 'ведмідь',
    'zebra': 'зебра',
    'giraffe': 'жирафа',
    'backpack': 'рюкзак',
    'umbrella': 'парасолька',
    'handbag': 'сумка',
    'tie': 'краватка',
    'suitcase': 'валіза',
    'laptop': 'ноутбук',
    'mouse': 'мишка',
    'remote': 'пульт',
    'keyboard': 'клавіатура',
    'cell phone': 'телефон',
    'book': 'книга',
    'bottle': 'пляшка',
    'cup': 'чашка',
    'spoon': 'ложка',
    'bowl': 'миска',
    'chair': 'стілець',
    'potted plant': 'вазон',
    'tv': 'телевізор'
}

#------------------ЗМІННІ------------------
object_counts: dict[str, int] = defaultdict(int)

trajectories: dict[str, deque[tuple[int, int]]] = defaultdict(
    lambda: deque(maxlen=MAX_TRAJECTORY_LEN)
)

last_sound_time: dict[str, float] = {}

roi_start: Optional[tuple[int, int]] = None
roi_end: Optional[tuple[int, int]] = None
drawing: bool = False
screenshot_timer: int = 0

#------------------ЗВУК------------------
def play_alert() -> None:
    threading.Thread(
        target=playsound,
        args=(SOUND_PATH,),
        daemon=True
    ).start()

#------------------ROI МИША------------------
def draw_roi(event: int, mx: int, my: int, _flags: int, _param: object) -> None:
    global roi_start, roi_end, drawing

    if event == cv2.EVENT_LBUTTONDOWN:
        roi_start = (mx, my)
        roi_end = None
        drawing = True

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        roi_end = (mx, my)

    elif event == cv2.EVENT_LBUTTONUP:
        roi_end = (mx, my)
        drawing = False

def get_normalized_roi() -> Optional[tuple[int, int, int, int]]:
    """Повертає нормалізований ROI"""
    if not roi_start or not roi_end:
        return None
    xs, xe = sorted([roi_start[0], roi_end[0]])
    ys, ye = sorted([roi_start[1], roi_end[1]])
    return xs, ys, xe, ye

#------------------МОДЕЛЬ------------------
model = torch.hub.load(
    'ultralytics/yolov5',
    'yolov5s',
    pretrained=True,
    trust_repo=True
)

#------------------ВІДЕО------------------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise RuntimeError("Камеру не відкрито")

cv2.namedWindow("YOLO Detection")
cv2.setMouseCallback("YOLO Detection", draw_roi)

log_file = open("detections.txt", "a", encoding="utf-8")
print("Керування:")
print("Q - Вихід")
print("S - Скріншот")
print("+/- - Поріг впевненості")
print("[/] - Поріг розміру")

#------------------ГОЛОВНИЙ ЦИКЛ------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    height, width = frame.shape[:2]
    frame_area = width * height

    results = model(frame)
    detections = results.pandas().xyxy[0]

    object_counts.clear()
    visible_classes = set(detections['name'].values)

    #очищення траєкторій
    for key in list(trajectories.keys()):
        if key not in visible_classes:
            trajectories[key].clear()

    roi = get_normalized_roi()

    for _, row in detections.iterrows():
        confidence = float(row['confidence'])
        if confidence < CONF_THRESHOLD:
            continue

        x1, y1 = int(row['xmin']), int(row['ymin'])
        x2, y2 = int(row['xmax']), int(row['ymax'])

        area = (x2 - x1) * (y2 - y1)
        if area / frame_area < MIN_OBJECT_PERCENT:
            continue

        center_x = (x1 + x2) // 2
        center_y = (y1 + y2) // 2

        #перевірка ROI
        if roi:
            x_min, y_min, x_max, y_max = roi
            if not (x_min <= center_x <= x_max and y_min <= center_y <= y_max):
                continue

        class_name = str(row['name'])
        label_ua = YOLO_CLASSES_UA.get(class_name, class_name)

        object_counts[label_ua] += 1
        trajectories[class_name].append((center_x, center_y))

        #звук з cooldown
        now = datetime.now().timestamp()
        if class_name in IMPORTANT_OBJECTS:
            last = last_sound_time.get(class_name, 0)
            if now - last > SOUND_COOLDOWN:
                play_alert()
                last_sound_time[class_name] = now

        #лог
        log_file.write(
            f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] "
            f"Виявлено: {label_ua} ({confidence * 100:.1f}%)\n"
        )

        #bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(
            frame,
            f"{label_ua} {confidence * 100:.1f}%",
            (x1, max(y1 - 5, 20)),
            cv2.FONT_HERSHEY_COMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

    #------------------ТРАЄКТОРІЇ------------------
    for points in trajectories.values():
        if len(points) >= 5:  # мінімум 5 точок
            pts = np.array(points, dtype=np.int32)
            cv2.polylines(frame, [pts], False, (255, 0, 0), 2)

    #------------------ROI(мал)------------------
    if roi:
        x_min, y_min, x_max, y_max = roi
        cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)

    #------------------СТАТИСТИКА------------------
    y = 25
    for obj, count in sorted(object_counts.items(), key=lambda x: -x[1])[:3]:
        cv2.putText(
            frame,
            f"{obj}: {count}",
            (10, y),
            cv2.FONT_HERSHEY_COMPLEX,
            0.7,
            (0, 255, 255),
            2
        )
        y += 30

    cv2.putText(
        frame,
        f"Conf: {int(CONF_THRESHOLD * 100)}% (+ / -) | Size: {int(round(MIN_OBJECT_PERCENT * 100))}%",
        (10, height - 20),
        cv2.FONT_HERSHEY_COMPLEX,
        0.7,
        (255, 255, 255),
        2
    )

    #повідомлення про скріншот
    if screenshot_timer > 0:
        cv2.putText(frame, "Screenshot Saved!", (width - 250, 40), cv2.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 0), 2)
        screenshot_timer -= 1

    cv2.imshow("YOLO Detection", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break

    elif key == ord('+'):
        CONF_THRESHOLD = min(0.95, CONF_THRESHOLD + 0.05)
    elif key == ord('='):
        CONF_THRESHOLD = min(0.95, CONF_THRESHOLD + 0.05)
    elif key == ord('-'):
        CONF_THRESHOLD = max(0.05, CONF_THRESHOLD - 0.05)
    elif key == ord(']'):
        MIN_OBJECT_PERCENT = min(0.5, MIN_OBJECT_PERCENT + 0.01)
    elif key == ord('['):
        MIN_OBJECT_PERCENT = max(0.0, MIN_OBJECT_PERCENT - 0.01)
    elif key == ord('s'):
        filename = f"screenshots/screen_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jpg"
        cv2.imwrite(filename, frame)
        screenshot_timer = 5
        print(f"Скріншот збережено: {filename}")



#------------------ЗАВЕРШЕННЯ------------------
log_file.close()
cap.release()
cv2.destroyAllWindows()
