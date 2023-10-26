import cv2
import torch
from tracker import *
import numpy as np
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)

cap=cv2.VideoCapture('videoo.mp4')


def POINTS(event, x, y, flags, param):
    if event == cv2.EVENT_MOUSEMOVE :  
        colorsBGR = [x, y]
        print(colorsBGR)
        

cv2.namedWindow('FRAME')
cv2.setMouseCallback('FRAME', POINTS)

tracker = Tracker()
while True:
    ret,frame=cap.read()
    frame=cv2.resize(frame,(1020,500))
    results=model(frame)
    #frame=np.squeeze(results.render())
    for index,row in results.pandas().xyxy[0].iterrows():
        x1=int(row['xmin'])
        y1=int(row['ymin'])
        x2=int(row['xmax'])
        y2=int(row['ymax'])
        b=str(row['name'])
        
        if 'car' in b:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(250,0,250),2)
            cv2.putText(frame,b,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(250,0,250),2)
            
        if 'bus' in b:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(255,0,0),2)
            cv2.putText(frame,b,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(255,0,0),2)
            
        if 'motorcycle' in b:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,250,0),2)
            cv2.putText(frame,b,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(0,250,0),2)
            
        if 'truck' in b:
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),2)
            cv2.putText(frame,b,(x1,y1),cv2.FONT_HERSHEY_PLAIN,1,(0,0,255),2)
        
        
    cv2.imshow('FRAME',frame)
    if cv2.waitKey(2)&0xFF==27:
        break
cap.release()
cv2.destroyAllWindows()
    
    
