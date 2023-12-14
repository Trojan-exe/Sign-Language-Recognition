import cv2
import mediapipe as mp
import time
import numpy as np
import handTrack as hd


def main():
    cap=cv2.VideoCapture(0)
    ptime=0
    ctime=0
    
    
    while True:
        sucess, img=cap.read()
        
        img=hd.handDetector().findHands(img)
        #img=hand.drawBoundingBoxes(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        img = cv2.flip(img, 1)
        cv2.putText(img,str(int(fps)),(18,70),cv2.FONT_HERSHEY_PLAIN,3
                    ,(255,0,255),3)
        img = cv2.resize(img, (600, 600))
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  
           break
    cap.release()  
    cv2.destroyAllWindows() 
    
if __name__ =="__main__":
    main()