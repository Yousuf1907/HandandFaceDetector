import cv2 as cv
import mediapipe as mp
import time

cap=cv.VideoCapture(0)
cTime=0
pTime=0

mpFaceDetection=mp.solutions.face_detection
mpDraw=mp.solutions.drawing_utils
faceDetection=mpFaceDetection.FaceDetection()

while True:
    success,img=cap.read()
    
    imgRGB=cv.cvtColor(img,cv.COLOR_BGR2RGB)
    results=faceDetection.process(imgRGB)
    # print(results)
    
    if results.detections:
        for id,detection in enumerate(results.detections):
            # mpDraw.draw_detection(img,detection)
            # print(id,detection)
            # print(detection.score)
            # print(detection.location_data.relative_bounding_box)
            bboxC=detection.location_data.relative_bounding_box
            # x,y,w,h=int(bboxC.x*img.shape[1]),int(bboxC.y*img.shape[0]),int(bboxC.width*img.shape[1]),int(bboxC.height*img.shape[0])
            ih,iw,ic=img.shape
            bbox=int(bboxC.xmin*iw), int(bboxC.ymin*ih),\
                int(bboxC.width*iw),int(bboxC.height*ih)
                
            cv.rectangle(img,bbox, (0,255,0),2)
            
            
    cTime=time.time()
    fps=1/(cTime-pTime)
    pTime=cTime
    
    cv.putText(img,f'FPS:{int(fps)}',(10,70),cv.FONT_HERSHEY_COMPLEX,1,(0,0,0),3)
    
    # cv.putText(img,f'{int(detection.score[0]*100)}%',(bbox[0],bbox[1]-20),cv.FONT_HERSHEY_DUPLEX,2,(255,255,255),1)
    cv.imshow("Image",img)
    
    cv.waitKey(10)
