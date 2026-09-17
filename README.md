# Edge Vision Alarm System

Smart security alarm built with Raspberry Pi 3, YOLOv8 object detection, and MQTT messaging.

## How It Works

1. Captures video frames from a camera or test video file.
2. Detects people using YOLOv8n via OpenCV DNN.
3. Publishes MQTT messages to topic `home/alarm`:
   - `DETECTAT` if a person is found.
   - `NONE` if no person is in frame.
4. Saves `detectie_persoana.jpg` with bounding boxes on successful detections.

## Setup & Run

1. Clone the repository:
```bash
git clone https://github.com/Miguel-Alessio/edge-vision-alarm.git
cd edge-vision-alarm
```

2. Start the MQTT broker:
```bash
docker compose up -d
```

3. Install requirements:
```bash
pip install -r requirements.txt --break-system-packages
```

4. Run the alarm:
```bash
python sistem_alarma.py
```
