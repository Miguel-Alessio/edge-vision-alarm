import cv2
import numpy as np
import paho.mqtt.client as mqtt
import time

MQTT_BROKER = "localhost"
MQTT_PORT = 1883
MQTT_TOPIC = "home/alarm"

client = mqtt.Client()
client.connect(MQTT_BROKER, MQTT_PORT, 60)

net = cv2.dnn.readNetFromONNX("camera-ai/yolov8n.onnx")
TARGET_CLASSES = {0: "PERSON"}

cap = cv2.VideoCapture("test.mp4")

last_sent_time = 0
COOLDOWN = 3.0

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

        blob = cv2.dnn.blobFromImage(frame, 1.0 / 255.0, (640, 640), swapRB=True, crop=False)
        net.setInput(blob)
        outputs = net.forward()

        if isinstance(outputs, list):
            outputs = outputs[0]
        out_matrix = np.array(outputs[0] if len(outputs.shape) == 3 else outputs).T

        detected = False
        for row in out_matrix:
            scores = row[4:]
            class_id = int(np.argmax(scores))
            confidence = float(scores[class_id])

            if confidence > 0.50 and class_id in TARGET_CLASSES:
                detected = True
                print(f"Detectat: {TARGET_CLASSES[class_id]} ({confidence:.2f})")
                break

        if detected:
            client.publish(MQTT_TOPIC, "DETECTAT")
            print("Trimis pe MQTT: DETECTAT")
            last_sent_time = current_time
        else:
            client.publish(MQTT_TOPIC, "NONE")
            print("Trimis pe MQTT: NONE")
            last_sent_time = current_time

        time.sleep(1.0)

except KeyboardInterrupt:
    cap.release()
    client.disconnect()
