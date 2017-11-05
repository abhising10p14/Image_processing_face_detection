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
	type thse in your terminal

	docker pull bamos/openface
	docker run -p 9000:9000 -p 8000:8000 -t -i bamos/openface /bin/bash
	cd /root/openface

	if any error occurs to enter into root mode, type:
	sudo -i



	Wait a second! I worte so many words which you gyuz might not be familliar about. 
	I would try my best to expalin each and every new term I write here
	So before talking about what is docker, we must know about containers, daemons

	Daemons-> They are the processes which run in the background and are not interactive. They have no controlling terminal.It runs in a multitasking OS like Unix.They perform certain actions at predefined times or in response to certain events. In *NIX, the names of daemons end in d.  
	
	Services - A service is a program which responds to requests from other programs over some inter-process communication mechanism. In Windows, daemons are called services.

	Process -A process is one or more threads of execution together with their shared set of resources, the most important of which are the address space and open file descriptors. A process creates an environment for these threads of execution which looks like they have an entire machine all to themselves: it is a virtual machine.

	Container-> It is like a sandbox for a process. There are multiple processes running in the OS.
	Now there is something similar to a sandbox calles as container which seperates the process from the other 
	processes. It is basically an isolated process which is running in a sandbox and that typically sees othe processes as an individual process.
	So what is a container image?
	An image is just a binary representation of a bunch of bits on a filesystem somewhere.In the same way as Virtual Machine Disk is a disk image.
	I am sure you would have isntalled or would have come thorugh an image file in your technical lifetime.
	So what are these image files actually built of? They are a simple compressed form which consist of scratch,busybox,sshd,aplications, other images in a hirearchy.
	Image is like a class of which you can create several instances and eaxh of these templates have containers 

	What is a docker file ? It is basically an environment in a text file. A docker file starts with
	From: 
	From is the parent image that the docker file is inherits from. Using this image file docker file configures several things to create a new image file. A docer file  is an starting point for an image.
	So this is the order Docker file to image to container. We can create an image file from containers also.

	What was the use of telling all these things?
	So that we can understand atleast 75% of all the upper level things being done here.SO what the above code is doing? It is actually downloading the openface library.
	Once you have run the above given codes type the followings:

	Step 1

	Make a folder called ./training-images/ inside the openface folder.	
	mkdir training-images

	Step 2
	mkdir ./training-images/will-ferrell/
	mkdir ./training-images/chad-smith/
	mkdir ./training-images/jimmy-fallon/


	Step 3
	Copy all your images of each person into the correct sub-folders. Make sure only one face appears in each image. There's no need to crop the image around the face. OpenFace will do that automatically.

	Step 4
	Run the openface scripts from inside the openface root directory:

	First, do pose detection and alignment:
	./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96

	This will create a new ./aligned-images/ subfolder with a cropped and aligned version of each of your test images.
	./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/

	After you run this, the ./generated-embeddings/ sub-folder will contain a csv file with the embeddings for each image.
	Third, train your face detection model:

	./demos/classifier.py train ./generated-embeddings/

	This will generate a new file called ./generated-embeddings/classifier.pkl. This file has the SVM model you'll use to recognize new faces.

	Step 5: Recognize faces!
	Get a new picture with an unknown face. Pass it to the classifier script like this:

	./demos/classifier.py infer ./generated-embeddings/classifier.pkl your_test_image.jpg

	You should get a prediction that looks like this:

		=== /test-images/will-ferrel-1.jpg ===
		Predict will-ferrell with 0.73 confidence.





	So this was a simple guide to understand how Face prediction is done. From here ownwards we can use
	this with a GUI or can make some alterations into it.
