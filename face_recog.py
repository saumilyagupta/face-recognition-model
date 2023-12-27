import cv2 as cv

# Gernal code to detect number of faces in a img 


img = cv.imread(r"C:\Users\saumi\OneDrive\Desktop\CODES\archive\database\Bollywood Actor Images\Bollywood Actor Images\adil_hussain\01d3ba3e5f.jpg") #img path
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
cv.imshow("gray",gray)

haarcascade_frontface_def =cv.CascadeClassifier("haarcascade_frontalface_default.xml")   #img Modal

facs_rec = haarcascade_frontface_def.detectMultiScale(gray, scaleFactor=3.0,minNeighbors=1)
print(f"Number of faces={len(facs_rec)}")

for (x,y,h,w) in facs_rec:
    cv.rectangle(img, (x,y), (x+h,y+w),(0,255,0),thickness=1)
cv.putText(img,f"{len(facs_rec)}",(25,55),cv.FONT_HERSHEY_COMPLEX,1.5,(201,221,234))
cv.imshow("img_", img)


cv.waitKey(0)

