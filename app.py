import streamlit as st
import os
import sys
from pathlib import Path
import tempfile

sys.path.append(str(Path(__file__).resolve().parent))
from src.utils import video_info, check_weights
from src.detect_video import VideoDetector


st.set_page_config(page_title='Coal & Rock Detection — Conveyor Belt AI', page_icon='⛏️', layout='wide')

st.sidebar.header('Model Settings')
weights_default = 'runs/train/rock_detection/weights/best.pt'
weights_path = st.sidebar.text_input('Weights path', value=weights_default)
conf_thres = st.sidebar.slider('Confidence threshold', 0.1, 0.9, 0.25)
iou_thres = st.sidebar.slider('IoU threshold', 0.1, 0.9, 0.45)

st.sidebar.markdown('**Model info**')
exists = check_weights(weights_path)
st.sidebar.write(f'Device: {"auto"}')
st.sidebar.write(f'Weights exist: {exists}')
if exists:
    try:
        size_mb = Path(weights_path).stat().st_size / (1024 * 1024)
        st.sidebar.write(f'Weights size: {size_mb:.1f} MB')
    except Exception:
        pass
else:
    st.sidebar.warning('No trained model found. Run python src/train.py first.')

st.title('Coal & Rock Detection — Conveyor Belt AI')
st.write('Upload a short conveyor-belt video to run detections using a trained YOLOv7 model.')

uploaded = st.file_uploader('Upload video', type=['mp4', 'avi', 'mov', 'mkv'])
if uploaded is not None:
    tdir = tempfile.mkdtemp()
    in_path = Path(tdir) / uploaded.name
    with open(in_path, 'wb') as f:
        f.write(uploaded.read())
    st.video(str(in_path))
    info = video_info(str(in_path))
    st.write(info)
    if st.button('Run Detection'):
        with st.spinner('Running detection...'):
            vd = VideoDetector(weights=weights_path, img_size=640, conf_thres=conf_thres, iou_thres=iou_thres)
            out = vd.detect_video(str(in_path))
        st.success('Detection finished')
        # show output video
        # find the annotated video in runs/detect/result
        out_dir = Path('runs') / 'detect' / 'result'
        video_file = None
        if out_dir.exists():
            for f in out_dir.iterdir():
                if f.suffix.lower() in ('.mp4', '.avi', '.mov', '.mkv'):
                    video_file = f
                    break
        if video_file:
            st.video(str(video_file))
        else:
            st.warning('Could not find output video. Check server logs.')

st.sidebar.markdown('---')
st.sidebar.markdown('By: **Md Shahedul Islam Khan**  
ML & AI Engineer, Research Assistant at the University of Newcastle  
[mdshahedulislamkhan.com](https://mdshahedulislamkhan.com) · [LinkedIn](https://linkedin.com/in/shahednpu)')
