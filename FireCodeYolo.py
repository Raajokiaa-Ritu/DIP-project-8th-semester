from ultralytics import YOLO
import cvzone
import cv2
import math
import playsound
import threading

model = YOLO('fireModel.pt') #model name
#cap = cv2.VideoCapture('f1.jpg') #---> image format must be = jpg
#cap = cv2.VideoCapture('testData2.mp4') #---> for video=mp4, 
cap = cv2.VideoCapture(0) #---> for real-time 0,1,2,......

# Reading the classes
classnames = ['fire']

while True:
    ret,frame = cap.read()
    #frame = cv2.flip(frame,1)
    frame = cv2.resize(frame,(720,480))
    result = model(frame,stream=True)

    # Getting bbox,confidence and class names informations to work with
    t = False
    for info in result:
        boxes = info.boxes
        for box in boxes:
            confidence = box.conf[0]
            confidence = math.ceil(confidence * 100)
            print("confidence: ",confidence)
            Class = int(box.cls[0])
            if confidence:
                t = True
            x1,y1,x2,y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1),int(y1),int(x2),int(y2)
            cv2.rectangle(frame,(x1,y1),(x2,y2),(0,0,255),5)
            cv2.putText(frame, 'FIRE DETECTED!', (30,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
            #cvzone.putTextRect(frame, f'fire', [x1, y1],scale=1.5,thickness=2)
            cvzone.putTextRect(frame, f'{classnames[Class]} {confidence}%', [x1 + 8, y1 + 100],scale=1.5,thickness=2)
    if t==False:
        cv2.putText(frame, 'No FIRE DETECTED!', (30,50), cv2.FONT_HERSHEY_PLAIN, 3, (0,255,0),2)
    cv2.imshow('frame',frame)
    if cv2.waitKey(10000) & 0xFF == ord('q'):
        break