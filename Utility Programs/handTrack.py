import cv2
import mediapipe as mp
import time
import numpy as np
import os
import string
class handDetector():
    def __init__(self):
      self.mpHands=mp.solutions.hands
      self.hands=self.mpHands.Hands(max_num_hands=1)
      self.mpDraw=mp.solutions.drawing_utils
     

    def findHands(self,img,roi,padd_amount = 15):
        imgRGB=cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result=self.hands.process(imgRGB)
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
                
                roi=img[y1:y2,x1:x2].copy()
                #roi= cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
                #blur = cv2.GaussianBlur(roi,(5,5),2)
                #th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY_INV,11,2)
                #ret, roi = cv2.threshold(th3, 70, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
                
        return img,roi
    
    def makeDir(self):
        if not os.path.exists("data"):
            os.makedirs("data")
        if not os.path.exists("data/train"):
            os.makedirs("data/train")
        if not os.path.exists("data/test"):
            os.makedirs("data/test")
        for i in string.ascii_uppercase:
            if not os.path.exists("data/train/" + i):
                os.makedirs("data/train/"+i)
            if not os.path.exists("data/test/" + i):
                os.makedirs("data/test/"+i)
    
    def drawCount(self,directory,frame):
       count = {
             'a': len(os.listdir(directory+"/A")),
             'b': len(os.listdir(directory+"/B")),
             'c': len(os.listdir(directory+"/C")),
             'd': len(os.listdir(directory+"/D")),
             'e': len(os.listdir(directory+"/E")),
             'f': len(os.listdir(directory+"/F")),
             'g': len(os.listdir(directory+"/G")),
             'h': len(os.listdir(directory+"/H")),
             'i': len(os.listdir(directory+"/I")),
             'j': len(os.listdir(directory+"/J")),
             'k': len(os.listdir(directory+"/K")),
             'l': len(os.listdir(directory+"/L")),
             'm': len(os.listdir(directory+"/M")),
             'n': len(os.listdir(directory+"/N")),
             'o': len(os.listdir(directory+"/O")),
             'p': len(os.listdir(directory+"/P")),
             'q': len(os.listdir(directory+"/Q")),
             'r': len(os.listdir(directory+"/R")),
             's': len(os.listdir(directory+"/S")),
             't': len(os.listdir(directory+"/T")),
             'u': len(os.listdir(directory+"/U")),
             'v': len(os.listdir(directory+"/V")),
             'w': len(os.listdir(directory+"/W")),
             'x': len(os.listdir(directory+"/X")),
             'y': len(os.listdir(directory+"/Y")),
             'z': len(os.listdir(directory+"/Z"))
             } 
       cv2.putText(frame, "a : "+str(count['a']), (10, 100), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "b : "+str(count['b']), (10, 110), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "c : "+str(count['c']), (10, 120), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "d : "+str(count['d']), (10, 130), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "e : "+str(count['e']), (10, 140), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "f : "+str(count['f']), (10, 150), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "g : "+str(count['g']), (10, 160), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "h : "+str(count['h']), (10, 170), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "i : "+str(count['i']), (10, 180), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "k : "+str(count['k']), (10, 190), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "l : "+str(count['l']), (10, 200), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "m : "+str(count['m']), (10, 210), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "n : "+str(count['n']), (10, 220), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "o : "+str(count['o']), (10, 230), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "p : "+str(count['p']), (10, 240), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "q : "+str(count['q']), (10, 250), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "r : "+str(count['r']), (10, 260), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "s : "+str(count['s']), (10, 270), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "t : "+str(count['t']), (10, 280), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "u : "+str(count['u']), (10, 290), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "v : "+str(count['v']), (10, 300), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "w : "+str(count['w']), (10, 310), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "x : "+str(count['x']), (10, 320), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "y : "+str(count['y']), (10, 330), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       cv2.putText(frame, "z : "+str(count['z']), (10, 340), cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 1)
       return frame,count
   
    def createImg(self,roi,directory,interrupt,count):
        
        if interrupt & 0xFF == ord('a'):
            cv2.imwrite(directory+'A/'+str(count['a'])+'.jpg', roi)
        if interrupt & 0xFF == ord('b'):
            cv2.imwrite(directory+'B/'+str(count['b'])+'.jpg', roi)
        if interrupt & 0xFF == ord('c'):
            cv2.imwrite(directory+'C/'+str(count['c'])+'.jpg', roi)
        if interrupt & 0xFF == ord('d'):
            cv2.imwrite(directory+'D/'+str(count['d'])+'.jpg', roi)
        if interrupt & 0xFF == ord('e'):
            cv2.imwrite(directory+'E/'+str(count['e'])+'.jpg', roi)
        if interrupt & 0xFF == ord('f'):
            cv2.imwrite(directory+'F/'+str(count['f'])+'.jpg', roi)
        if interrupt & 0xFF == ord('g'):
            cv2.imwrite(directory+'G/'+str(count['g'])+'.jpg', roi)
        if interrupt & 0xFF == ord('h'):
            cv2.imwrite(directory+'H/'+str(count['h'])+'.jpg', roi)
        if interrupt & 0xFF == ord('i'):
            cv2.imwrite(directory+'I/'+str(count['i'])+'.jpg', roi)
        if interrupt & 0xFF == ord('j'):
            cv2.imwrite(directory+'J/'+str(count['j'])+'.jpg', roi)
        if interrupt & 0xFF == ord('k'):
            cv2.imwrite(directory+'K/'+str(count['k'])+'.jpg', roi)
        if interrupt & 0xFF == ord('l'):
            cv2.imwrite(directory+'L/'+str(count['l'])+'.jpg', roi)
        if interrupt & 0xFF == ord('m'):
            cv2.imwrite(directory+'M/'+str(count['m'])+'.jpg', roi)
        if interrupt & 0xFF == ord('n'):
            cv2.imwrite(directory+'N/'+str(count['n'])+'.jpg', roi)
        if interrupt & 0xFF == ord('o'):
            cv2.imwrite(directory+'O/'+str(count['o'])+'.jpg', roi)
        if interrupt & 0xFF == ord('p'):
            cv2.imwrite(directory+'P/'+str(count['p'])+'.jpg', roi)
        if interrupt & 0xFF == ord('q'):
            cv2.imwrite(directory+'Q/'+str(count['q'])+'.jpg', roi)
        if interrupt & 0xFF == ord('r'):
            cv2.imwrite(directory+'R/'+str(count['r'])+'.jpg', roi)
        if interrupt & 0xFF == ord('s'):
            cv2.imwrite(directory+'S/'+str(count['s'])+'.jpg', roi)
        if interrupt & 0xFF == ord('t'):
            cv2.imwrite(directory+'T/'+str(count['t'])+'.jpg', roi)
        if interrupt & 0xFF == ord('u'):
            cv2.imwrite(directory+'U/'+str(count['u'])+'.jpg', roi)
        if interrupt & 0xFF == ord('v'):
            cv2.imwrite(directory+'V/'+str(count['v'])+'.jpg', roi)
        if interrupt & 0xFF == ord('w'):
            cv2.imwrite(directory+'W/'+str(count['w'])+'.jpg', roi)
        if interrupt & 0xFF == ord('x'):
            cv2.imwrite(directory+'X/'+str(count['x'])+'.jpg', roi)
        if interrupt & 0xFF == ord('y'):
            cv2.imwrite(directory+'Y/'+str(count['y'])+'.jpg', roi)
        if interrupt & 0xFF == ord('z'):
            cv2.imwrite(directory+'Z/'+str(count['z'])+'.jpg', roi)
        
         

def main():  
    
    cap=cv2.VideoCapture(0)
    ptime=0
    ctime=0
    hand=handDetector()
    hand.makeDir()
   #minValue = 70
    mode = 'train'
    directory = 'data/'+mode+'/'
    while True:
        sucess, img=cap.read()
        img = cv2.flip(img, 1)
        roi=img[210:101,202:260].copy()
        img,roi=hand.findHands(img,roi)
        cv2.imshow("test",roi)
        #img=hand.drawBoundingBoxes(img)
        ctime=time.time()
        fps=1/(ctime-ptime)
        ptime=ctime
        
        cv2.putText(img,str(int(fps)),(18,70),cv2.FONT_HERSHEY_PLAIN,3
                    ,(255,0,255),3)
        img,count=hand.drawCount(directory, img)
        interrupt=cv2.waitKey(10)
        #roi = img[y1:y2, x1:x2].copy()
        hand.createImg(roi,directory,interrupt,count) 
        img = cv2.resize(img, (800, 600))
        cv2.imshow("image", img)
        if cv2.waitKey(1) & 0xFF == 27:  
           break
           
    cap.release()  
    cv2.destroyAllWindows() 
    
    
    
    
  
if __name__ == "__main__":  
    main()  