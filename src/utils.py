import os
import yaml
import cv2
import torch
from pathlib import Path
import matplotlib.pyplot as plt


def get_device():
    if torch.cuda.is_available():
        return 'cuda:0'
    # check mps (Apple Silicon)
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def check_weights(path: str) -> bool:
    p = Path(path)
    if not p.exists():
        print(f"Weights not found at {path}")
        return False
    if p.stat().st_size < 1024 * 1024:
        print(f"Weights file {path} looks too small (<1MB)")
        return False
    return True


def check_dataset(yaml_path: str) -> dict:
    p = Path(yaml_path)
    if not p.exists():
        raise FileNotFoundError(f"Data YAML not found: {yaml_path}")
    with open(p) as f:
        info = yaml.safe_load(f)
    base = p.parent.parent
    for k in ('train', 'val', 'test'):
        if k in info:
            path = base / info[k]
            if not path.exists():
                print(f"Warning: {k} path does not exist: {path}")
    return info


def video_info(path: str) -> dict:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {path}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 0.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    duration = total / fps if fps > 0 else 0.0
    cap.release()
    return {
        'width': width,
        'height': height,
        'fps': fps,
        'total_frames': total,
        'duration_seconds': duration,
    }


def count_labels(labels_dir: str) -> dict:
    p = Path(labels_dir)
    counts = {}
    if not p.exists():
        return counts
    for f in p.glob('*.txt'):
        for line in f.read_text().splitlines():
            if not line.strip():
                continue
            parts = line.split()
            cls = int(parts[0])
            counts[f'class_{cls}'] = counts.get(f'class_{cls}', 0) + 1
    return counts


def plot_label_distribution(labels_dir: str, class_names: list, save_path: str = None):
    counts = {}
    for i, n in enumerate(class_names):
        counts[n] = 0
    d = count_labels(labels_dir)
    for k, v in d.items():
        idx = int(k.split('_')[1])
        if idx < len(class_names):
            counts[class_names[idx]] = counts.get(class_names[idx], 0) + v

    names = list(counts.keys())
    vals = [counts[n] for n in names]
    plt.figure(figsize=(6, 4))
    plt.bar(names, vals)
    plt.title('Label distribution')
    plt.ylabel('Bounding boxes')
    if save_path:
        plt.savefig(save_path, bbox_inches='tight')
        print(f"Saved distribution plot to {save_path}")
    else:
        plt.show()
