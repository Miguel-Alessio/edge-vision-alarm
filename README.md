# Edge Vision Alarm System

IoT-based security alarm pipeline performing object detection on an edge device (Raspberry Pi 3) and triggering real-time alerts via MQTT.

## Features
- Lightweight inference using YOLOv8n (ONNX format) executed through OpenCV DNN.
- MQTT publish-subscribe architecture with Eclipse Mosquitto broker.
- Bounding box rendering and alert logging on positive detections.

## Setup

1. Start the MQTT broker:
	bash
	docker compose up -d
2. Install dependencies:
	bash
	pip install -r requirements.txt --break-system-packages
3. Run the alarm system:
	bash
	python sistem_alarma.py
