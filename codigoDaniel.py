import torch
import cv2
import numpy as np
import pandas


model = torch.hub.load("ultralytics/yolov5", "custom", 
                       path = "C:/Users/JONATHAN GONZALEZ/OneDrive/Documents/deteccionDeObjetos DanielM/best(5).pt")


cap=cv2.VideoCapture(3)

while (cap.isOpened()):
    r,frame=cap.read()
    detec = model(frame)
    cv2.imshow("Detector", np.squeeze(detec.render()))
    if r==True:
        cv2.imshow('Video',frame)
        
        if cv2.waitKey(25) &0xFF == ord('e'):
            break
    else:
        break

cap.release()
cv2.destroyAllWindows()