import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
import torch


@dataclass
class TrainConfig:
    weights: str = 'yolov7/yolov7.pt'
    cfg: str = 'yolov7/cfg/training/yolov7.yaml'
    data: str = 'config/rock_data.yaml'
    epochs: int = 50
    batch_size: int = 16
    img_size: int = 640
    device: str = ''
    project: str = 'runs/train'
    name: str = 'rock_detection'
    workers: int = 4
    patience: int = 10


def detect_device():
    if torch.cuda.is_available():
        return '0'
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return 'mps'
    return 'cpu'


def run_training(cfg: TrainConfig):
    cwd = Path(__file__).resolve().parent.parent
    yolov7_dir = cwd / 'yolov7'
    orig_cwd = Path.cwd()
    device = cfg.device or detect_device()

    # Resolve paths to absolute to run train.py from inside yolov7/
    def _resolve(p):
        if not p:
            return ''
        pp = Path(p)
        return str(pp) if pp.is_absolute() else str((cwd / p).resolve())

    weights_arg = _resolve(cfg.weights)
    cfg_arg = _resolve(cfg.cfg)
    # prefer dataset-provided data.yaml if present
    data_yaml_candidate = cwd / 'data' / 'rock_dataset' / 'data.yaml'
    if data_yaml_candidate.exists():
        data_arg = str(data_yaml_candidate.resolve())
    elif (cwd / cfg.data).exists():
        data_arg = _resolve(cfg.data)
    else:
        data_arg = _resolve(cfg.data)

    cmd = [sys.executable, 'train.py',
           '--weights', weights_arg,
           '--cfg', cfg_arg,
           '--data', data_arg,
           '--epochs', str(cfg.epochs),
           '--batch-size', str(cfg.batch_size),
           '--img', str(cfg.img_size),
           '--device', device,
           '--project', cfg.project,
           '--name', cfg.name,
           '--workers', str(cfg.workers),
           '--cache',
           '--exist-ok']

    print('Starting training with command:', ' '.join(cmd))
    try:
        if yolov7_dir.exists():
            os.chdir(yolov7_dir)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b''):
            if not line:
                break
            sys.stdout.write(line.decode(errors='ignore'))
        proc.wait()
    finally:
        os.chdir(orig_cwd)

    best = Path(cfg.project) / cfg.name / 'weights' / 'best.pt'
    print(f"Training finished. Best weights (expected): {best}")


if __name__ == '__main__':
    cfg = TrainConfig()
    run_training(cfg)
