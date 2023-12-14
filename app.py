import streamlit as st
from streamlit_webrtc import webrtc_streamer
import av
import cv2
import mediapipe as mp
import numpy as np
import glob

st.set_page_config(page_title="Sign Language Recognition", page_icon = "logo2.png", layout = "wide", initial_sidebar_state = "expanded") 
st.title("Sign Language Recognition")

if 'images' not in st.session_state:
    st.session_state.images = glob.glob('images/*.jpg')

with st.sidebar:
    st.title("Gestures For Characters")
    rows = [st.columns(3) for _ in range(9)]
    cols = [column for row in rows for column in row]
    for col, Image in zip(cols, st.session_state.images):
        col.image(Image)

class originalVideoProcessor:
    def recv(self, frame):
        img=frame.to_ndarray(format="bgr24")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

class VideoProcessor:
    def __init__(self):  
      self.mpHands=mp.solutions.hands
      self.hands=self.mpHands.Hands(max_num_hands=1)
      self.mpDraw=mp.solutions.drawing_utils

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result=self.hands.process(imgRGB)
        padd_amount = 15
        if result.multi_hand_landmarks:
            landmarks = [] 
            for handLms in result.multi_hand_landmarks:
                
                for id, landmark in enumerate(handLms.landmark):
                    height, width, _ = img.shape
                    landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                         (landmark.z * width)))
                self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                x_coordinates = np.array(landmarks)[:,0]
                y_coordinates = np.array(landmarks)[:,1]
                x1  = int(np.min(x_coordinates) - padd_amount)
                y1  = int(np.min(y_coordinates) - padd_amount)
                x2  = int(np.max(x_coordinates) + padd_amount)
                y2  = int(np.max(y_coordinates) + padd_amount)
                cv2.rectangle(img, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
        return av.VideoFrame.from_ndarray(img, format="bgr24")


col1, col2 = st.columns(2)
with col2:
    ctx = webrtc_streamer(
        key="example",
        video_processor_factory=VideoProcessor,
        rtc_configuration={  
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )

with col1:
    ctx2 = webrtc_streamer(
        key="example2",
        video_processor_factory=originalVideoProcessor,
        rtc_configuration={  
            "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
        }
    )


col1.metric("Present Character", value = "A")
col1.subheader("Complete Sentence")
col1.text("Welcome! to the world of Sign Language")

