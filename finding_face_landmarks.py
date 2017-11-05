import sys 
import dlib
from skimage import io

'''
	In the previous step we isolated the faces from the rest part of a picture 
	but as we can see from the result that there are certain faces which 
	remained undetected in the picture.
	Why this occured ?
	The problem was that faces turned in different direction look totally 
	different to a computer and thus it was unable to detect those faces which were 
	either turned or far away from being detected.

	To overcome this problem what do we need to do ??
	We will try to warp each picture so that the eyes and lips are always in the 
	sample place in the image. This will make 
	it a lot easier for us to compare faces in the next steps

	For this we will be using an algorithm called as Face landmark 
	estimation 
	Before landmark detection  you need to do face detection 

	So what is landamark detection basically ?
	It chooses around 68 points(landmarks) on the face. These points exists 
	on every face - the top of the chin the outside edge of each eye, the inner edge of each eyebrow, etc.
	 Then we will train a machine learning algorithm to be able to find these 68 specific points on any face
	
	Now that we know where the face is actually located inside the pic we can 
	rotate ,scale and shear the image so that the eyes and mouth are centered as best as possible. 
	There are certain frameworks which provide a fancy 3D wrap around the face 
	whic determines where the face is turned twards but we are not going to 
	implement that.
	We are only going to use basic image transformations like rotation and scale that 
	preserve parallel lines (called affine transformations)
	Now no matter how the face is turned, we are able to center the eyes and mouth are in roughly 
	the same position in the image. This will make our next step a lot more accurate.

	Either we can build up our own face detection model or could just download 
	it from "http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2"
	 there are other pre-trained model I have downloaded second such model from 
	 http://dlib.net/files/mmod_human_face_detector.dat.bz2
	Personaly , as newbie with simple system I would prefer the pre-built model

	'''
predictor_model = "shape_predictor_68_face_landmarks.dat"


# creating a HOG face detector  using built-in dlib class 

face_detector = dlib.get_frontal_face_detector()

# Creating a face pose predictor from the built-in dlib class 
face_pose_predictor  = dlib.shape_predictor(predictor_model)

win = dlib.image_window()

# give the local address of the image file 
file_name = "pic6.jpg"

#load the image into an array 
image = io.imread(file_name)

#Now run the HOG face detector on the image data 
detected_faces =  face_detector(image,1)
print("Found {} faces in the image file {}".format(len(detected_faces),file_name))

# display the desktop window with image
win.set_image(image)

#Loop through each face we detected in th eimage 

'''
The enumerate() function adds a counter to an iterable.

So for each element in cursor, a tuple is produced with (counter, element); 
the for loop binds that to row_number and row, respectively.

'''
for i, face_rect in enumerate(detected_faces):
	# Detected faces are returned as an object with the coordinates 
	# of the top, left, right and bottom edges
	print("- Face #{} found at Left: {} Top: {} Right: {} Bottom: {}".format(i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
	# draw a box around each face we found
	win.add_overlay(face_rect)

	# Get the the face's pose
	pose_landmarks = face_pose_predictor(image, face_rect)

	# Draw the face landmarks on the screen.
	win.add_overlay(pose_landmarks)

dlib.hit_enter_to_continue()