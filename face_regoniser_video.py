import cv2 as cv
import numpy as np
import os

haarcascade_frontface_def =cv.CascadeClassifier("C:/Users/saumi/OneDrive/Desktop/CODES/opencv/opencv-4.x/data/haarcascades/haarcascade_frontalface_default.xml")

DIR=r'C:\Users\saumi\OneDrive\Desktop\CODES\archive\database\Bollywood Actor Images\Bollywood Actor Images'

people=[]
for i in os.listdir(DIR):
    people.append(i)
# print(people)

face_recon = cv.face.LBPHFaceRecognizer_create()
face_recon.read("face_trained.yml")

# vid = cv.VideoCapture(0)
vid = cv.VideoCapture(r"Untitled video - Made with Clipchamp.mp4") #tast data


while(True):

    rate,frame = vid.read()
    gary_frame = cv.cvtColor(frame,cv.COLOR_BGR2GRAY)
    cv.imshow("raw", frame)
    face_rect = haarcascade_frontface_def.detectMultiScale(gary_frame,1.1,4)

    for (x,y,w,h) in face_rect:
        face_crop = gary_frame[y:y+h,x:x+w]

        lable, confidance = face_recon.predict(face_crop)
        cv.putText(frame,f"{people[lable]}\n {int(confidance)}",(x,y+h+10),cv.FONT_HERSHEY_DUPLEX,0.3,(0,255,255) )
        cv.rectangle(frame, (x,y), (x+w,y+h),(0,255,0),thickness=1 )
    
    cv.imshow("output", frame)



    if cv.waitKey(1) & (0xFF == ord('q')):
        break 

cv.waitKey(0)    
