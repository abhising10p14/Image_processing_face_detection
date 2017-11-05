'''
	So here I am going to explain how to build a face recognization model of your own.
	It took me around 3-4 days of filling the required module directories and put the required openface 
	models and the lua in models in correct directories.So I will explain each and ever thing
	so that you can save your time and don't haeve to increase the number of tabs of your browser and finally 
	go for a long nap :) .
	What to do?
	First open the face_recognization_dib.py. I have explained each and every thing and have tried my best to explain 
	each and every thing
	You can either use dlib library or the opencv library for face detection but in my case, I got better 
	result from dlib. Though you should run both the files face_recognization_dlib_faces.py and
	face_detector_opencv.py 
	What these 2 above given files are doing ?
	They are detecting humean faces oout of the whole picture provided to them. Note the fact that opencv only 
	works for .jpg and .png images so better try to provide only these formats for training purpose

	After the face detection face you need to allign the image so that there are certain images which are not 
	easily recognizable by the computers.So inorder to allign the images provided in the input we will run the 
	facd_allignment.py which will the save the alligned images in our local directory

	After the face detection phase what you need to do is to find the landmarks on the face 
	for this open and run the finding_face_landmarks.py
	What this program does ?
	It actually devides the face found in the previous step and sets a set of landmarks around the faces 
	These landmarks are nothing but a set of pre-defined points which are present in every humean face
	If you run the finding_face_landmarks.py you will see what these landmarks actually look like.
	
	Encoding faces :
	Now comes the main part of the problem i.e tellibg the faces apart.
	For a second, think in simplest terms how you are gonna do this. 
	The simplest way would be compare the input face with all the pre-loaded images and that which is most close 
	yo the given inoput would be the answer.
	The same approach is used with the help of deep neural network which do the job of face recognization in 
	a very efficient way.
	But here of training the network to recognize pictures objects , we
	are going to train it to generate 128 measurements for each face. 
	The training process works by looking at 3 face images at a time:

    1.Load a training face image of a known person
    2.Load another picture of the same known person
    3.Load a picture of a totally different person
	
	Then the algorithm looks at the measurements it is currently generating for each of those three images.
	It then tweaks the neural network slightly so that it makes sure the measurements it generates for twp pics
	of the same person are slightly closer while making sure the measurements for different person  are 
	slightly further apart.
	After running for several iterations the Nureal Network generates 128 measurements for each person.
	This process is called as Embedding.
	This exact approach was started by Google in 2015.

	This process of training a convolutional neural network to output face embeddings 
	requires a lot of data and computer power.

	But once the network has been trained, it can generate measurements for any face, even ones it has never seen before!
	So this step only needs to be done once. Lucky for us, the fine folks at OpenFace already did this and they published 
	several trained networks which we can directly use.
	So all we need to do ourselves is run our face images through their pre-trained network to get the 128 measurements 
	for each face.

	Now comes the final step of face recognization i.e passing a random pic and let you ML model(which can be any
	classifier model , SVM in this case ) decide who is this person. 
	All we need to do is train a classifier that can take in the measurements from a new test image
	and tells which known person is the closest match. Running this classifier takes milliseconds. The
	result of the classifier is the name of the person!


	So  lets start this process of face recognization :

	Before you start :
	you must have OpenFace and Dlib installed 
	you can either install them manually 
	or follow these steps :
	docker pull bamos/openface
	docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
	cd /root/openface
	
'''