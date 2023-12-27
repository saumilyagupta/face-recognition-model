import cv2 as cv
import numpy as np
import os
# import face_recognition

haarcascade_frontface_def =cv.CascadeClassifier("haarcascade_frontalface_default.xml")


DIR=r'Bollywood Actor Images\Bollywood Actor Images'  # training data base

people=[]
feature =[]
lables =[]
for i in os.listdir(DIR):
    people.append(i)
 
print(people) 
def train_model():
    i=0
    for person in people:
        path = os.path.join(DIR, person)
        lable= people.index(person)
        j=0
        for img in os.listdir(path):
            img_path = os.path.join(path,img)
            img_temp = cv.imread(img_path)
            gray_temp= cv.cvtColor(img_temp, cv.COLOR_BGR2GRAY)

            face_rect = haarcascade_frontface_def.detectMultiScale(gray_temp, scaleFactor=1.1, minNeighbors=4)

            print(f"{i}{person},{j}")
            for (x,y,w,h) in face_rect:
                face_crop = gray_temp[y:y+h,x:x+w]
                feature.append(face_crop)
                lables.append(lable)
            j=j+1 
            i=i+1       

train_model()
print("Taring done--------\n")

feature= np.array(feature, dtype='object')
lables= np.array(lables)

# i=0
# for img in feature:
#     cv.imshow(f"{i}",img)
#     i=i+1

# face_recon = cv.face.LBPHFaceRecognizer()
face_recon = cv.face.LBPHFaceRecognizer_create()
# face_recon = cv.face.


face_recon.train(feature,lables)
# cv.face.LBPHFaceRecognizer.train(feature,lables)
face_recon.save("face_trained.yml")
# cv.face.LBPHFaceRecognizer.save("face_trained.yml")
# cv.face

np.save("features.npy", feature)
np.save("lables.npy", lables)

cv.waitKey(0)