import os
import sys
import subprocess
import re
from dataclasses import dataclass
from pathlib import Path


@dataclass
class EvalConfig:
    weights: str = 'runs/train/rock_detection/weights/best.pt'
    data: str = 'config/rock_data.yaml'
    img_size: int = 640
    batch: int = 16
    device: str = ''
    task: str = 'test'


def run_evaluation(cfg: EvalConfig):
    cwd = Path(__file__).resolve().parent.parent
    yolov7_dir = cwd / 'yolov7'
    orig_cwd = Path.cwd()
    cmd = [sys.executable, 'test.py',
           '--weights', cfg.weights,
           '--data', cfg.data,
           '--img-size', str(cfg.img_size),
           '--batch-size', str(cfg.batch),
           '--device', cfg.device or '',
           '--task', cfg.task,
           '--verbose']
    result = {'mAP_0.5': None, 'mAP_0.5:0.95': None}
    try:
        if yolov7_dir.exists():
            os.chdir(yolov7_dir)
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        for line in iter(proc.stdout.readline, b''):
            if not line:
                break
            s = line.decode(errors='ignore')
            sys.stdout.write(s)
            # try to parse mAP lines
            m = re.search(r"mAP@0.5\s*:\s*([0-9.]+)", s)
            if m:
                try:
                    result['mAP_0.5'] = float(m.group(1))
                except Exception:
                    pass
            m2 = re.search(r"mAP@0.5:0.95\s*:\s*([0-9.]+)", s)
            if m2:
                try:
                    result['mAP_0.5:0.95'] = float(m2.group(1))
                except Exception:
                    pass
        proc.wait()
    finally:
        os.chdir(orig_cwd)
    print('Evaluation results:', result)
    return result


if __name__ == '__main__':
    cfg = EvalConfig()
    run_evaluation(cfg)
