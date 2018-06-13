import cv2
import numpy as np
import os

def reconnaissance_faciale(image_jpg):
    
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read('/home/pi/Projet/FacialRecognitionProject/trainer/trainer.yml')#Attention, ici il faut indiquer le chemin du dossier dans lequel vous allez enregistrer le fichier .yml qui correspond aux matrices déduies des visages enregistrés
    cascadePath = "/home/pi/Projet/haarcascades/haarcascade_frontalface_default.xml" #ATTENTION, ici il faut changer le chemin du dossier dans lequel vous avez telecharge le fichier haarcascade auparavant
    faceCascade = cv2.CascadeClassifier(cascadePath);
    font = cv2.FONT_HERSHEY_SIMPLEX
    information=[]
    id=0
    i=1

    while True:
        img = cv2.imread(image_jpg)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        faces = faceCascade.detectMultiScale(gray, scaleFactor = 1.3, minNeighbors=5)
        for(x,y,w,h) in faces:
            if i<6:
                cv2.rectangle(img, (x,y), (x+w,y+h), (0,255,0), 2)
                information.append([x,y])
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])
                if (confidence<80):
                    id = "Connu"
                    confidence = " {0}%".format(round(100 - confidence))
                else:
                    id = "unknown"
                    confidence = " {0}%".format(round(100-confidence))
                    
                information.append(id)
                i=i+1               
        break
    print(information)
    cv2.destroyAllWindows()
    return(information)
