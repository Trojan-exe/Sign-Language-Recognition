import streamlit as st
from streamlit_webrtc import webrtc_streamer
import cv2
import mediapipe as mp
import av
import numpy as np
from collections import Counter
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications import imagenet_utils
import time
import queue
from tensorflow.keras.models import model_from_json
import glob
model=load_model('new_model_saurabh.h5')

st.set_page_config(page_title="Sign Language Recognition", page_icon = "logo2.png", layout = "centered", initial_sidebar_state = "expanded") 
st.title("Sign Language Recognition")

if 'images' not in st.session_state:
    st.session_state.images = glob.glob('images/*.jpg')
    
with st.sidebar:
    st.title("ASL CHARACTERS")
    rows = [st.columns(3) for _ in range(9)]
    cols = [column for row in rows for column in row]
    for col, Image in zip(cols, st.session_state.images):
        col.image(Image)

class VideoProcessor:
    def __init__(self):
      self.mpHands=mp.solutions.hands
      self.hands=self.mpHands.Hands(max_num_hands=1)
      self.mpDraw=mp.solutions.drawing_utils
      self.result_queue = queue.Queue()

    def recv(self, frame):
        print("Inside recv function")
        img = frame.to_ndarray(format="bgr24")
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result=self.hands.process(imgRGB)
        padd_amount = 35
        if result.multi_hand_landmarks:
            landmarks = [] 
            for handLms in result.multi_hand_landmarks:
                
                for id, landmark in enumerate(handLms.landmark):
                    height, width, _ = img.shape
                    landmarks.append((int(landmark.x * width), int(landmark.y * height),
                                         (landmark.z * width)))
                #self.mpDraw.draw_landmarks(img, handLms,self.mpHands.HAND_CONNECTIONS)
                x_coordinates = np.array(landmarks)[:,0]
                y_coordinates = np.array(landmarks)[:,1]
                x1  = int(np.min(x_coordinates) - padd_amount)
                y1  = int(np.min(y_coordinates) - padd_amount)
                x2  = int(np.max(x_coordinates) + padd_amount)
                y2  = int(np.max(y_coordinates) + padd_amount)
                cv2.rectangle(img, (x1, y1), (x2, y2), (155, 0, 255), 3, cv2.LINE_8)
                roi=img[y1:y2,x1:x2].copy()
                img_name="1.png"
                roi = cv2.resize(roi, (224,224))
                roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                blur = cv2.GaussianBlur(roi,(5,5),2)
                th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                ret, save_img = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                cv2.imwrite(img_name, save_img)
                img1 = image.load_img('1.png', target_size=(224, 224))
                x = image.img_to_array(img1)   
                x=x/255
                x = np.expand_dims(x, axis=0)
                preds = model.predict(x)
                preds=np.argmax(preds, axis=1)
                
                if preds==0:
                    self.result_queue.put("A")
                    cv2.putText(img,"A",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                            ,(255,0,255),3)
                elif preds==1:
                    self.result_queue.put("B")
                    cv2.putText(img,"B",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==2:
                    self.result_queue.put("C")
                    cv2.putText(img,"C",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==3:
                    self.result_queue.put("D")
                    cv2.putText(img,"D",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==4:
                    self.result_queue.put("E")
                    cv2.putText(img,"E",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==5:
                    self.result_queue.put("F")
                    cv2.putText(img,"F",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==6:
                    self.result_queue.put("G")
                    cv2.putText(img,"G",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==7:
                    self.result_queue.put("H")
                    cv2.putText(img,"H",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==8:
                    self.result_queue.put("I")
                    cv2.putText(img,"I",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==9:
                    self.result_queue.put("J")
                    cv2.putText(img,"J",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==10:
                    self.result_queue.put("K")
                    cv2.putText(img,"K",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==11:
                    self.result_queue.put("L")
                    cv2.putText(img,"L",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==12:
                    self.result_queue.put("M")
                    cv2.putText(img,"M",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==13:
                    self.result_queue.put("N")
                    cv2.putText(img,"N",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==14:
                    self.result_queue.put("O")
                    cv2.putText(img,"O",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==15:
                    self.result_queue.put("P")
                    cv2.putText(img,"p",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==16:
                    self.result_queue.put("Q")
                    cv2.putText(img,"Q",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==17:
                    self.result_queue.put("R")
                    cv2.putText(img,"R",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==18:
                    self.result_queue.put("S")
                    cv2.putText(img,"S",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                            ,(255,0,255),3)
                elif preds==19:
                    self.result_queue.put("T")
                    cv2.putText(img,"T",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==20:
                    self.result_queue.put("U")
                    cv2.putText(img,"U",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==21:
                    self.result_queue.put("V")
                    cv2.putText(img,"V",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==22:
                    self.result_queue.put("W")
                    cv2.putText(img,"W",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==23:
                    self.result_queue.put("X")
                    cv2.putText(img,"X",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==24:
                    self.result_queue.put("Y")
                    cv2.putText(img,"Y",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                elif preds==25:
                    self.result_queue.put("Z")
                    cv2.putText(img,"Z",(18,70),cv2.FONT_HERSHEY_PLAIN,3
                                ,(255,0,255),3)
                else:
                    self.result_queue.put(" ")
        print("Exiting recv function")
        return av.VideoFrame.from_ndarray(img, format="bgr24")

ctx = webrtc_streamer(
    key="example",
    video_processor_factory=VideoProcessor,
    rtc_configuration={  
        "iceServers": [
            {"urls": ["stun:stun.services.mozilla.com"]},
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    },
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("Present Character")
character_placeholder = st.empty()
st.write("Word")
word_placeholder = st.empty()

if st.checkbox("Show the detected labels", value=True):
        if ctx.state.playing:       
            word = ""
            count=0;
            word1=""
            while True:
                if ctx.video_processor:
                    try:
                        result = ctx.video_processor.result_queue.get(
                            timeout=1.0
                        )
                    except queue.Empty:
                        result = None
                    
                    character_placeholder.write(result)
                    if count < 20:
                        if result != None:
                            word = word + result
                        
                        count=count+1
                    else:
                        word = Counter(word)
                        w = max(word, key = word.get)
                        word1=word1+w;
                        word_placeholder.write(word1)
                        count=0
                        word=""
                else:
                    break
