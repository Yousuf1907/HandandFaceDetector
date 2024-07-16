import cv2 as cv
import numpy as np
import mediapipe as mp
import time
import random

# I used these color to randomly change landmark colors
r=random.randint(0,255)
g=random.randint(0,255)
b=random.randint(0,255)

r2=random.randint(0,255)
g2=random.randint(0,255)
b2=random.randint(0,255)

mphands=mp.solutions.hands
hands=mphands.Hands()

mpFace=mp.solutions.face_detection
faceDetection=mpFace.FaceDetection()

mpDraw=mp.solutions.drawing_utils
pTime=0

cap=cv.VideoCapture(0)
# fps = cap.get(cv.CAP_PROP_FPS)  Built-in method to calculate the Frames rate per second.   

while True:
    success,img=cap.read()
    img=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    
    results=hands.process(img)
    if results.multi_hand_landmarks:
        for landmark in results.multi_hand_landmarks:
            mpDraw.draw_landmarks(img,landmark,mphands.HAND_CONNECTIONS)
    
    face_results = faceDetection.process(img)
    if face_results.detections:
        for id,detection in enumerate(face_results.detections):
            mpDraw.draw_detection(img,detection)
            
            bboxC=detection.location_data.relative_bounding_box
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw), int(bboxC.ymin*ih),\
                int(bboxC.width*iw),int(bboxC.height*ih)
                
            cv.rectangle(img,bbox, (0,255,0),2)
    
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    
    cv.putText(img,f"FPS:{int(fps)}",(10,70),cv.FONT_HERSHEY_PLAIN,1,(r,g,b),2)
    cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_DUPLEX,2,(r2,g2,b2),1)
    
    cv.imshow("Tracker",img) 
    if cv.waitKey(20) & 0xFF==ord('q'):
        break   
        
cap.release()
cv.destroyAllWindows()