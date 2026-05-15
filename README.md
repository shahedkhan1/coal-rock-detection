# Coal & Rock Detection on Conveyor Belts

Conveyor belt monitoring in mining operations is traditionally a manual job — someone watches a monitor and flags anomalies. When a belt is moving tonnes of material at speed, that's both unreliable and expensive. This project replaces that with a YOLOv7 model trained to detect and classify rocks and coal chunks in real time from conveyor footage.

Built during my time doing ML engineering work on industrial computer vision systems. The main challenges were variable lighting in underground/industrial environments, motion blur at high belt speeds, and rocks of wildly different sizes partially occluding each other.

![Python](https://img.shields.io/badge/Python-3.10+-blue) ![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red) ![YOLOv7](https://img.shields.io/badge/YOLO-v7-purple) ![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green) ![Streamlit](https://img.shields.io/badge/Demo-Streamlit-ff4b4b)

---

## Pipeline

![Architecture](architecture.svg)

---

## Results

| Metric | Value |
|---|---|
| mAP@0.5 | *run `python src/evaluate.py` to populate* |
| mAP@0.5:0.95 | *run `python src/evaluate.py` to populate* |
| Inference speed | *depends on hardware* |
| Training epochs | 50 |
| Input resolution | 640 × 640 |

---

## Setup

```bash
git clone https://github.com/shahedkhan1/coal-rock-detection.git
cd coal-rock-detection

python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Run the setup script to clone YOLOv7 and download pretrained weights:

```bash
python setup.py
```

---

## Dataset

Download from Roboflow Universe (free account, no API key needed for manual download):

```bash
python data/download_dataset.py
```

Or place your own YOLO-format export at `data/rock_dataset/` with the standard layout:
```
data/rock_dataset/
├── train/images/   train/labels/
├── valid/images/   valid/labels/
└── test/images/    test/labels/
```

Then update `config/rock_data.yaml` to match your class names.

---

## Usage

**Smoke test** (verify the pipeline works before committing to a full run):
```bash
python run_smoke.py
```

**Full training** (50 epochs, saves best checkpoint to `runs/train/rock_detection/weights/best.pt`):
```bash
python src/train.py
```

**Evaluate** on the test set:
```bash
python src/evaluate.py
```

**Run on a video**:
```bash
python src/detect_video.py \
  --weights runs/train/rock_detection/weights/best.pt \
  --source path/to/conveyor_video.mp4
```

Output saved to `yolov7/runs/detect/result/`.

**Streamlit demo**:
```bash
streamlit run app.py
```

Upload a video, adjust confidence and IoU thresholds from the sidebar, run detection.

---

## Notes

- Works on CPU, Apple Silicon (MPS), and CUDA — auto-detected at runtime
- PyTorch 2.6+ users: `torch.load` calls are patched to use `weights_only=False` for full checkpoint compatibility
- If dataset paths are wrong, check that `data/rock_dataset/data.yaml` exists and the `train/valid/test` paths inside it are correct relative to that file

---

## Project structure

```
coal-rock-detection/
├── setup.py                  # Clone YOLOv7, download pretrained weights
├── run_smoke.py              # 1-epoch smoke test
├── app.py                    # Streamlit demo
├── config/
│   └── rock_data.yaml        # Dataset config
├── data/
│   ├── download_dataset.py   # Roboflow download
│   └── augment.py            # Offline augmentation
├── src/
│   ├── train.py              # Training wrapper
│   ├── evaluate.py           # mAP evaluation
│   ├── detect_video.py       # Video inference pipeline
│   └── utils.py              # Shared helpers
└── yolov7/                   # Vendored YOLOv7 source
```

---

## Author

**Md Shahedul Islam Khan** — ML & AI Engineer, Research Assistant at the University of Newcastle.  
[mdshahedulislamkhan.com](https://mdshahedulislamkhan.com) · [LinkedIn](https://linkedin.com/in/shahednpu) · [GitHub](https://github.com/shahedkhan1)
