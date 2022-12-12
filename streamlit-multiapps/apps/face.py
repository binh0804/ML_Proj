from io import StringIO
from pathlib import Path
import streamlit as st
import cv2
from yolov5.test2 import detect
import os
import numpy as np
import pandas as pd
import av
import argparse
from PIL import Image
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import tempfile
from glob import glob

def get_subdirs(b="."):
    """
    Returns all sub-directories in a specific Path
    """
    result = []
    for d in os.listdir(b):
        bd = os.path.join(b, d)
        if os.path.isdir(bd):
            result.append(bd)
    return result


def get_detection_folder():
    """
    Returns the latest folder in a runs\detect
    """
    return max(get_subdirs(os.path.join("runs", "detect")), key=os.path.getmtime)


def app():
    # with open('static/style.css','r') as f:
    #     st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    st.title("Face detection")
    st.write("## 1. Dataset :cat: :duck: :dog: ")
    st.write("### Context")
    st.write(
        "It is important to detect humans on footage of CCTV, so, let us use this dataset to train a neural network to do it"
    )
    st.write("### Content")
    st.write(
        "Dataset contains CCTV footage images(as indoor as outdoor), a half of them w humans and a half of them is w/o humans. "
    )
    st.write("### Sources of dataset:")
    st.write("1) cctv footage from youtube;")
    st.write("2) open indoor images dataset;")
    st.write("3) footage from my cctv.")
    cols = st.columns(2)  # number of columns in each row! = 2
    # first column of the ith row
    cols[0].image(
        "https://user-images.githubusercontent.com/26833433/127574988-6a558aa1-d268-44b9-bf6b-62d4c605cc72.jpg",
        use_column_width=True,
    )
    cols[1].image(
        "https://user-images.githubusercontent.com/26833433/127574988-6a558aa1-d268-44b9-bf6b-62d4c605cc72.jpg",
        use_column_width=True,
    )

    st.write("## 2. References :bank: :bangbang: ")
    st.write("  1. Python")
    st.write("  2. YoloV5")
    st.write("  3. OpenCV")
    st.write("## 3. Demo :factory: ")
    st.write("")
    st.write("")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--weights",
        nargs="+",
        type=str,
        default="weights/yolov5s.pt",
        help="model.pt path(s)",
    )
    parser.add_argument("--source", type=str, default="data/images", help="source")
    parser.add_argument(
        "--img-size", type=int, default=640, help="inference size (pixels)"
    )
    parser.add_argument(
        "--conf-thres", type=float, default=0.35, help="object confidence threshold"
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="IOU threshold for NMS"
    )
    parser.add_argument(
        "--device", default="", help="cuda device, i.e. 0 or 0,1,2,3 or cpu"
    )
    parser.add_argument("--view-img", action="store_true", help="display results")
    parser.add_argument("--save-txt", action="store_true", help="save results to *.txt")
    parser.add_argument(
        "--save-conf", action="store_true", help="save confidences in --save-txt labels"
    )
    parser.add_argument(
        "--nosave", action="store_true", help="do not save images/videos"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by class: --class 0, or --class 0 2 3",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS"
    )
    parser.add_argument("--augment", action="store_true", help="augmented inference")
    parser.add_argument("--update", action="store_true", help="update all models")
    parser.add_argument(
        "--project", default="runs/detect", help="save results to project/name"
    )
    parser.add_argument("--name", default="exp", help="save results to project/name")
    parser.add_argument(
        "--exist-ok",
        action="store_true",
        help="existing project/name ok, do not increment",
    )
    opt = parser.parse_args()

    source = ("Hình ảnh", "Video")
    source_index = st.sidebar.selectbox(
        "Chọn đầu vào", range(len(source)), format_func=lambda x: source[x]
    )


    if source_index == 0:
        uploaded_file = st.file_uploader("Load an image", type=["png", "jpeg", "jpg"])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text="Đang tải ảnh lên..."):
                st.image(uploaded_file)
                picture = Image.open(uploaded_file)


                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.jpeg')


                picture = picture.save(tfile.name)
                opt.source = tfile.name

        else:
            is_valid = False
    elif source_index == 1:
        uploaded_file = st.file_uploader("Load a video", type=["mp4"])
        if uploaded_file is not None:
            is_valid = True
            with st.spinner(text="Đang tải video lên..."):
                st.video(uploaded_file)

 
                tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
                tfile.write(uploaded_file.getbuffer())

                opt.source = tfile.name


        else:
            is_valid = False
    else:
        is_valid = True
        # def process(img):
        #     opt.source = img
        class VideoProcessor:
            def recv(self, frame):
                img = frame.to_ndarray(format="bgr24")
                img = cv2.flip(img, 1)
                return av.VideoFrame.from_ndarray(img, format="bgr24")

    if is_valid:
        if source_index == 2:
            RTC_CONFIGURATION = RTCConfiguration(
                {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
            )
            webrtc_ctx = webrtc_streamer(
                key="WYH",
                mode=WebRtcMode.SENDRECV,
                rtc_configuration=RTC_CONFIGURATION,
                media_stream_constraints={"video": True, "audio": False},
                video_processor_factory=VideoProcessor,
                async_processing=True,
            )
        else:
            if st.button("Loading..."):
                processed = False
                MSG_POKER = detect(opt,Image.open(temp_frame))
                if source_index == 0:
                    with st.spinner(text="Preparing Images"):
                        for img in os.listdir(get_detection_folder()):
                            print(str(Path(f"{get_detection_folder()}") / img))
                            st.image(str(Path(f"{get_detection_folder()}") / img))
                        st.balloons()
                        processed = False

                elif source_index == 1:

                    with st.spinner(text="Preparing Video"):
                        for vid in glob(get_detection_folder() + '/*.mp4'):
                            print(vid)
                            with open(vid, 'r') as f:
                                st.video(vid)
                        st.balloons()
                        processed = True
