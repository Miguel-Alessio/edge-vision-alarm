import os
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import math
import time

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "home/alarm"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

net = cv2.dnn.readNetFromONNX("camera-ai/yolov8n.onnx")

cap = cv2.VideoCapture("test.mp4")

if not cap.isOpened():
    print("EROARE: Nu se poate deschide test.mp4")
    exit(1)

FPS = cap.get(cv2.CAP_PROP_FPS)
if FPS <= 0 or math.isnan(FPS):
    FPS = 25.0
FRAME_SKIP = max(1, int(round(FPS * 0.2)))

CONF_THRESHOLD = 0.35
NMS_THRESHOLD = 0.40

frame_idx = 0
total_oameni = 0
max_in_val = 0
cadre_goale_consecutive = 0
PRAG_CADRE_GOALE = 5

last_persoane = -1
last_total = -1

def format_letterbox(img):
    h, w = img.shape[:2]
    max_dim = max(h, w)
    padded = np.zeros((max_dim, max_dim, 3), dtype=np.uint8)
    padded[0:h, 0:w] = img
    return padded, max_dim / 640.0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            if max_in_val > 0:
                total_oameni += max_in_val
            print(f"Final clip. Persoane totale numarate: {total_oameni}")
            break

        frame_idx += 1
        if frame_idx % FRAME_SKIP != 0:
            continue

        padded, scale = format_letterbox(frame)
        blob = cv2.dnn.blobFromImage(padded, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        if isinstance(outputs, list):
            outputs = outputs[0]
        out_matrix = np.array(outputs[0] if len(outputs.shape) == 3 else outputs).T

        boxes = []
        confidences = []

        for row in out_matrix:
            person_score = float(row[4])
            if person_score >= CONF_THRESHOLD:
                cx = row[0] * scale
                cy = row[1] * scale
                w = row[2] * scale
                h = row[3] * scale

                x1 = int(cx - w / 2)
                y1 = int(cy - h / 2)
                boxes.append([x1, y1, int(w), int(h)])
                confidences.append(person_score)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESHOLD, NMS_THRESHOLD)
        persoane_in_cadru = len(indices)

        if persoane_in_cadru > 0:
            cadre_goale_consecutive = 0
            if persoane_in_cadru > max_in_val:
                max_in_val = persoane_in_cadru
        else:
            cadre_goale_consecutive += 1
            if cadre_goale_consecutive >= PRAG_CADRE_GOALE and max_in_val > 0:
                total_oameni += max_in_val
                print(f">>> Val incheiat: +{max_in_val} persoane. Total pana acum: {total_oameni}")
                max_in_val = 0

        total_afisat = total_oameni + max_in_val
        stare_schimbata = (persoane_in_cadru != last_persoane) or (total_afisat != last_total)

        if persoane_in_cadru > 0:
            saved_frame = frame.copy()
            for idx in indices:
                i = idx[0] if isinstance(idx, (list, np.ndarray)) else idx
                bx, by, bw, bh = boxes[i]
                conf = confidences[i]

                x1 = max(0, bx)
                y1 = max(0, by)
                x2 = min(frame.shape[1], bx + bw)
                y2 = min(frame.shape[0], by + bh)

                cv2.rectangle(saved_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.putText(saved_frame, f"OM ({conf:.2f})", (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            info_text = f"In cadru: {persoane_in_cadru} | Total estimat: {total_afisat}"
            cv2.putText(saved_frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            output_path = os.path.abspath("detectie_persoana.jpg")
            cv2.imwrite(output_path, saved_frame)
            client.publish(MQTT_TOPIC, "DETECTAT")
            
            if stare_schimbata:
                print(f"In cadru: {persoane_in_cadru} | Total estimat: {total_afisat}")
        else:
            client.publish(MQTT_TOPIC, "NONE")
            if stare_schimbata:
                print(f"Trimis pe MQTT: NONE | Total contorizati: {total_afisat}")

        if stare_schimbata:
            last_persoane = persoane_in_cadru
            last_total = total_afisat

        time.sleep(0.1)

except KeyboardInterrupt:
    pass

cap.release()
client.disconnect()
