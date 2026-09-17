import os
import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "home/alarm"

client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
client.connect(MQTT_BROKER, MQTT_PORT, 60)

net = cv2.dnn.readNetFromONNX("camera-ai/yolov8n.onnx")
TARGET_CLASSES = {0: "PERSON"}

cap = cv2.VideoCapture("test.mp4")

last_sent_time = 0
COOLDOWN = 2.0
CONF_THRESHOLD = 0.40

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            continue

        current_time = time.time()
        if current_time - last_sent_time < COOLDOWN:
            time.sleep(0.05)
            continue

        h_orig, w_orig = frame.shape[:2]

        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        if isinstance(outputs, list):
            outputs = outputs[0]
        out_matrix = np.array(outputs[0] if len(outputs.shape) == 3 else outputs).T

        detected = False
        saved_frame = frame.copy()

        for row in out_matrix:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > CONF_THRESHOLD and class_id in TARGET_CLASSES:
                detected = True
                
                cx = int(row[0] * (w_orig / 640.0))
                cy = int(row[1] * (h_orig / 640.0))
                w = int(row[2] * (w_orig / 640.0))
                h = int(row[3] * (h_orig / 640.0))
                
                x1 = max(0, int(cx - w / 2))
                y1 = max(0, int(cy - h / 2))
                x2 = min(w_orig, int(cx + w / 2))
                y2 = min(h_orig, int(cy + h / 2))

                cv2.rectangle(saved_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                label = f"{TARGET_CLASSES[class_id]}: {confidence:.2f}"
                cv2.putText(saved_frame, label, (x1, max(25, y1 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                break

        if detected:
            output_path = os.path.abspath("detectie_persoana.jpg")
            success = cv2.imwrite(output_path, saved_frame)
            client.publish(MQTT_TOPIC, "DETECTAT")
            print(f"Detectat & Salvat imaginea la: {output_path} (Succes scriere: {success})")
            last_sent_time = current_time
        else:
            client.publish(MQTT_TOPIC, "NONE")
            print("Trimis pe MQTT: NONE")
            last_sent_time = current_time

        time.sleep(0.5)

except KeyboardInterrupt:
    cap.release()
    client.disconnect()
