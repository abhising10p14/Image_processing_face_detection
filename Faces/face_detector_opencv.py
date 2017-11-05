import numpy as np
import cv2

front_face_cascade = cv2.CascadeClassifier('/home/abhishek/ml/Image_processing/Faces/opencv/data/haarcascades/haarcascade_frontalface_alt.xml')
side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')
eye_cascade = cv2.CascadeClassifier('/home/abhishek/ml/Image_processing/Faces/opencv/data/haarcascades/haarcascade_eye.xml')
#face_cascade.empty()
img = cv2.imread('pic4.jpg')

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#Now we find the faces in the image. If faces are found, it returns the positions of detected faces as Rect(x,y,w,h). Once we get these locations, we can create a ROI for the face and apply eye detection on this ROI (since eyes are always on the face !!! ).
front_faces  = front_face_cascade.detectMultiScale(gray, 1.1, 5)

side_faces = side_face_cascade.detectMultiScale(gray,1.1,5)


for (x,y,w,h) in front_faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

# for the side faces 
for (x,y,w,h) in side_faces:
    cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = img[y:y+h, x:x+w]
    eyes = eye_cascade.detectMultiScale(roi_gray)
    for (ex,ey,ew,eh) in eyes:
        cv2.rectangle(roi_color,(ex,ey),(ex+ew,ey+eh),(0,255,0),2)

print(len(front_faces))
print(len(side_faces))
cv2.imshow('img',img)
cv2.waitKey(0)
cv2.destroyAllWindows()


'''
	So which one to use OpenCv or dlib?
	what is the differencd between them ?
	The core difference is:

    DLib is a C++ library/toolkit that contains machine learning algorithms, including computer vision.
    OpenCV is a C/C++ library of functions dealing with real-time computer vision.
    but the question again remains unanswered. which one should be used? As compared to the speed, both take nearly same amount of time 
    though there are lot of errors when it comes to opencv
    
'''