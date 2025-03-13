from turtle import color
import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture(0)

mpHands=mp.solutions.hands
hands=mpHands.Hands()

mpDraw=mp.solutions.drawing_utils

pTime,cTime=0,0
fps = cap.get(cv.CAP_PROP_FPS)

while True:
    success,img=cap.read()
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=hands.process(imgRGB)
    print(results.multi_hand_landmarks)
    
    if results.multi_hand_landmarks:
        for handLms in results.multi_hand_landmarks:
            for id,lm in enumerate(handLms.landmark):
                # print(id,lm)
                h,w,c=img.shape
                cy=int(lm.y*h)
                cx= int(lm.x*w)
                print(id,cx,cy)
                cv.circle(img,(cx,cy),8,(1,2,3),cv.FILLED)
   
    pTime=cTime
    
    cv.putText(img,str(int(fps)), (10,70),cv.FONT_HERSHEY_COMPLEX,2,(255,0,255),3)        
    cv.imshow("Image",img)
    
    if cv.waitKey(20) & 0xFF==ord('q'):
        break
        
cap.release()
cv.destroyAllWindows()
