import sys
import dlib
import numpy as np
import cv2  # importing opencv which is c++ library for image processing 
from skimage import io  # skimage is a python library for image processing 
'''The histogram of oriented gradients (HOG) is a feature descriptor used in computer vision and image processing for the purpose of object detection. 
The technique counts occurrences of gradient orientation in localized portions of an image. 
This method is similar to that of edge orientation histograms, scale-invariant feature transform descriptors, and shape contexts, 
but differs in that it is computed on a dense grid of uniformly spaced cells and uses overlapping local contrast normalization for
 improved accuracy.
 It actually calculates graident relatively to the nearby cells of the blocks of the image.

https://www.youtube.com/watch?v=Dl5lPdoCXi8

Gradient points in the direction of most rapid increase in intensity 
Edge detection -> the point where there isa change in the intensity 
				of the image . i.e from high to low or vice versa
		https://www.youtube.com/watch?v=V2z_x80xPzI
		
		Ramp edege and sharp edege 


'''

'''
How image is stored as data ? 

Resolution -> Dimension by which we can measure how many pixels are on the screen 
Density -> Allows you t store the same resolution image on different screen size 
A single pixel consists of three components of RED,GREEN and Blue colors
which range from 0-255 .	0 is very dark 
and 255 would be very bright 
Triplets of these values together can roduce a single pixel
for exapmle (255,255,255) will give you a value of pixel whose color is WHITE

An Image file whether is a JPEG,GIF,PNG contains millions of these 
RGB triplets. All these data is stored as a bit i.e '0' or '1'
Each color chanel of a RGB is represented by 8 bits i.e 1 BYTEas 
it ranges from 0-255.For example the RGB value of the color TURQUOISE are 
(64,224,208) respectively.
A computer would store this as
R:01000000
G:11100000
B:11010000
When the same data is stored in Hexadecimal :
40 E0 D0    which is way lot shorter

How do we change the tone of colors?
We just pass the RGB data of the image and pass it to a mapping 
function which changes the tone of the color as described in that function 

'''

file_name = "pic2.jpg"
# Create a HOG face detector using the built-in dlib class
face_detector = dlib.get_frontal_face_detector()
#face_detector= dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
# for side face detection 
side_face_cascade = cv2.CascadeClassifier('haarcascade_profileface.xml')

win = dlib.image_window() # A window used to display the image 

# Load the image into an array
image = io.imread("pic2.jpg")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Run the HOG face detector on the image data.
# The result will be the bounding boxes of the faces in our image.
detected_faces = face_detector(image, 1)
side_faces = side_face_cascade.detectMultiScale(gray,1.1,5)
print("I found {} front faces in the file {}".format(len(detected_faces), file_name))

print("I found {}side  faces in the file {}".format(len(side_faces), file_name))

# Open a window on the desktop showing the image
win.set_image(image)

# Loop through each face we found in the image
for i, face_rect in enumerate(detected_faces):

	# Detected faces are returned as an object with the coordinates 
	# of the top, left, right and bottom edges
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))

	# Draw a box around each face we found
	win.add_overlay(face_rect)

# for the side faces 
for (x,y,w,h) in side_faces:
    cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,0),2)
    roi_gray = gray[y:y+h, x:x+w]
    roi_color = image[y:y+h, x:x+w]

# Wait until the user hits <enter> to close the window	        
dlib.hit_enter_to_continue()