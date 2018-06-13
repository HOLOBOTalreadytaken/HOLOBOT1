import cv2
import numpy as np
from PIL import Image
import os

#Fonction qui nous permet de recuperer les photos prises et enregistrées auparavant (fonction que l'on réutilise dans notre fonction principale d'enregistrement)
def getImagesAndLabels(path):
    detector = cv2.CascadeClassifier('/home/pi/Projet/haarcascades/haarcascade_frontalface_default.xml') #ATTENTION, ici il faut changer le chemin du dossier dans lequel vous avez telecharge le fichier haarcascade auparavant
    imagePaths = [os.path.join(path,f) for f in os.listdir(path)]
    faceSamples=[]
    ids=[]
    for imagePath in imagePaths:
        PIL_img = Image.open(imagePath).convert('L')
        img_numpy = np.array(PIL_img, 'uint8')
        id = int(os.path.split(imagePath)[-1].split(".")[1])
        faces = detector.detectMultiScale(img_numpy)
        for (x,y,w,h) in faces:
            faceSamples.append(img_numpy[y:y+h, x:x+w])
            ids.append(id)
    return faceSamples, ids

def enregistrement_database(id_liste):
    face_detector = cv2.CascadeClassifier('/home/pi/Projet/haarcascades/haarcascade_frontalface_default.xml') #ATTENTION, ici il faut changer le chemin du dossier dans lequel vous avez telecharge le fichier haarcascade auparavant
    path='/home/pi/Projet/FacialRecognitionProject/dataset' #ATTENTION, ici il faut changer le chemin du dossier dans lequel vous enregistrez les photos
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    cam=cv2.VideoCapture(0)
    cam.set(3, 640)
    cam.set(4, 480)
    count=0
    while(True):
        ret, img = cam.read()
        img =cv2.flip(img, -1)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = face_detector.detectMultiScale(gray, 1.3, 5)
        
        for(x,y,w,h) in faces:
            cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 2)
            count +=1
            
            cv2.imwrite("/home/pi/Projet/FacialRecognitionProject/dataset/User."+ str(id_liste)+'.'+str(count)+".jpg",gray[y:y+h,x:x+w]) #Attention, ici il faut aussi changer le chemin du dossier dans lequel on enregistre les photos
            cv2.imshow('image', img)
        k=cv2.waitKey(100) & 0xff
        if k == 27:
            break
        elif count >=15:
            break
    cam.release()
    cv2.destroyAllWindows()
    faces,ids=getImagesAndLabels(path)
    recognizer.train(faces, np.array(ids))
    recognizer.write('/home/pi/Projet/FacialRecognitionProject/trainer/trainer.yml') #Attention, ici il faut indiquer le chemin du dossier dans lequel vous allez enregistrer le fichier .yml qui correspond aux matrices déduies des visages enregistrés
    print("\n {0} faces traines. Exiting Program".format(len(np.unique(ids))))
    print(id_liste)
    return(id_liste)
