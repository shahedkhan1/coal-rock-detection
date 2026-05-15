import os
import sys
import subprocess
from pathlib import Path
import argparse
import cv2
from typing import Optional


class VideoDetector:
    def __init__(self, weights: str, img_size: int = 640, conf_thres: float = 0.25, iou_thres: float = 0.45, device: str = ''):
        self.weights = weights
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device

    def _detect_device(self):
        import torch
        if self.device:
            return self.device
        if torch.cuda.is_available():
            return '0'
        if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            return 'mps'
        return 'cpu'

    def detect_video(self, source: str, output_path: Optional[str] = None, show_preview: bool = False) -> str:
        cwd = Path(__file__).resolve().parent.parent
        yolov7_dir = cwd / 'yolov7'
        orig_cwd = Path.cwd()
        device = self._detect_device()
        # Resolve relative paths to absolute so detect.py (run from yolov7/) can find them
        def _resolve(p):
            pp = Path(p)
            return str(pp) if pp.is_absolute() else str((cwd / p).resolve())

        weights_arg = _resolve(self.weights)
        source_arg = _resolve(source)

        cmd = [sys.executable, 'detect.py',
               '--weights', weights_arg,
               '--source', source_arg,
               '--img-size', str(self.img_size),
               '--conf-thres', str(self.conf_thres),
               '--iou-thres', str(self.iou_thres),
               '--device', device,
               '--project', 'runs/detect',
               '--name', 'result',
               '--exist-ok']
        if show_preview:
            cmd.append('--view-img')

        print('Running detection:', ' '.join(cmd))
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

        # detect.py writes to yolov7/{project}/{name}/
        out_dir = yolov7_dir / Path('runs') / 'detect' / 'result'
        if out_dir.exists():
            # prefer video files if present
            for c in out_dir.iterdir():
                if c.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                    return str(c.resolve())
            return str(out_dir.resolve())
        return str((cwd / 'runs' / 'detect' / 'result').resolve())

    def detect_webcam(self):
        return self.detect_video(source='0', show_preview=True)


def draw_results_summary(output_dir: str):
    p = Path(output_dir)
    labels_dir = p / 'labels'
    counts = {}
    total_frames = 0
    if labels_dir.exists():
        for f in labels_dir.glob('*.txt'):
            total_frames += 1
            for line in f.read_text().splitlines():
                if not line.strip():
                    continue
                cls = int(line.split()[0])
                counts[cls] = counts.get(cls, 0) + 1

    print(f"Total frames processed (labels files): {total_frames}")
    for cls, c in counts.items():
        print(f"class_{cls} detections: {c}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True)
    parser.add_argument('--source', required=True)
    parser.add_argument('--conf', type=float, default=0.25)
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    vd = VideoDetector(weights=args.weights, conf_thres=args.conf)
    out = vd.detect_video(args.source, output_path=args.output)
    print('Detection output:', out)
    # attempt to summarize
    summary_dir = Path('runs') / 'detect' / 'result'
    if summary_dir.exists():
        draw_results_summary(str(summary_dir))


if __name__ == '__main__':
    main()
