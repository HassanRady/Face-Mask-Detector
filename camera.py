import cv2
import numpy as np

from model import initialize

face = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
font = cv2.FONT_HERSHEY_SIMPLEX
model = initialize()


class VideoCamera(object):
    
    def __init__(self):
        self.video = cv2.VideoCapture(0)
        
    def __del__(self):
        self.video.release()
        
    def get_frame(self):
        _, fr = self.video.read()
        img = cv2.cvtColor(fr, cv2.COLOR_BGR2RGB)
        m = 0
#         img = cv2.resize(img, (224+m, 224+m))

        faces = face.detectMultiScale(img, scaleFactor=1.3, minNeighbors=3)

        for (x, y, w, h) in faces:
            fc = img[y-m:y+h+m, x-m:x+w+m]
            pred = model.predict(fc)
            cv2.putText(fr, pred, (x, y), font, 1, (255, 255, 0), 2)
            cv2.rectangle(fr,(x,y),(x+w,y+h),(255,0,0),2)
#             break
            
        _, jpeg = cv2.imencode('.jpg', fr)
        return jpeg.tobytes()