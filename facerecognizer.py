# Face Recognition module
import numpy as np
import cv2

class FaceRecognizer:

    def load_model(self):
        self.face_cascade = cv2.CascadeClassifier("facerecognizer.xml")
    
    #detect faces in images
    def detect_faces(self, image_data):
        # convert binary data
        nparr = np.fromstring(image_data, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        # detect faces coordinates
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.3, 4)
        results = []
        for (x,y,w,h) in faces:
            element = {'x' : str(x), 'y' : str(y), 'width' : str(w), 'height' : str(h)}
            results.append(element)
        return results
