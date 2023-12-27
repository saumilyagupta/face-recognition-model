import cv2 as cv
import numpy as np
import os

haarcascade_frontface_def =cv.CascadeClassifier("haarcascade_frontalface_default.xml")

DIR=r'test_data' #test database

people=[]
for i in os.listdir(r"Bollywood Actor Images\Bollywood Actor Images"):  #training Database
    people.append(i)
# print(people)
# features= np.load('features.npy')
# lables = np.load(r"C:\Users\saumi\OneDrive\Desktop\CODES\opencv\code_base_opencv\Face_rec\lables.npy")

face_recon = cv.face.LBPHFaceRecognizer_create()
face_recon.read("face_trained.yml")

# img = cv.imread(r"Faces\val\elton_john\2.jpg")
def fun(path):

    # img = cv.imread(r"C:\Users\saumi\OneDrive\Desktop\CODES\opencv\code_base_opencv\Face_rec\Faces\val\ben_afflek\2.jpg")
    img=cv.imread(path)

    gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)

    face_rect = haarcascade_frontface_def.detectMultiScale(gray, 1.1,4)

    for (x,y,w,h) in face_rect:
        face_crop = gray[y:y+h,x:x+w]

        lable, confidance = face_recon.predict(face_crop)
        cv.rectangle(img, (x,y), (x+w,y+h), (0,255,0),thickness=1)
        cv.putText(img,f"{people[lable]}\n {int(confidance)}",(x,y+h+10),cv.FONT_HERSHEY_DUPLEX,0.3,(0,255,255) )
    
    cv.imshow("output", img)


# for img in os.listdir(DIR):
    # path = os.path.join(DIR,img)
    # fun(r"path")
fun(r"Bollywood Actor Images\Bollywood Actor Images")
cv.waitKey(0)