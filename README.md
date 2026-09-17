# Edge Vision Alarm System

Smart security alarm built with Raspberry Pi 3, YOLOv8 object detection, and MQTT messaging.

## Quick Start (Copy-Paste Setup)

Run these commands in order. Everything needed (including packages and a sample test video with people) is downloaded automatically:

```bash
# 1. Clone the repository and enter the project folder
git clone https://github.com/Miguel-Alessio/edge-vision-alarm.git
cd edge-vision-alarm

# 2. Download sample test video
curl -L -o test.mp4 "https://github.com/intel-iot-devkit/sample-videos/raw/master/people-detection.mp4"

# 3. Install the terminal image viewer
sudo apt update && sudo apt install -y chafa

# 4. Start the MQTT broker
docker compose up -d

# 5. Install Python dependencies
pip install -r requirements.txt --break-system-packages

# 6. Run the detection system
python3 sistem_alarma.py
```

Wait until you see the following line printed in the terminal:
```text
Detectat & Salvat imaginea detectie_persoana.jpg
```

Press **`Ctrl + C`** to stop the script, then view the detected person directly in your terminal:

```bash
chafa detectie_persoana.jpg
```

---

## What Just Happened?

1. OpenCV loaded `test.mp4` and processed the frames using `camera-ai/yolov8n.onnx`.
2. When a person appeared with confidence $> 0.50$, the system:
   - Published `DETECTAT` to the MQTT topic `home/alarm`.
   - Saved `detectie_persoana.jpg` with a green bounding box around the detected person.
3. When no person was present, it published `NONE`.
