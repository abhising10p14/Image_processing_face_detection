'''
	Now comesthe task of training the models for face embedding 
	what is face embedding?
	The training process works by looking at 3 face images at a time:

    Load a training face image of a known person
    Load another picture of the same known person
    Load a picture of a totally different person
    Then the algorithm looks at the measurements it is currently generating for each of those three images. 
    It then tweaks the neural network slightly so that it makes sure the measurements it generates for #1 and #2.
    After repeating this step millions of times for millions of images of thousands of different people, the neural
    network learns to reliably generate 128 measurements for each person. Any ten different pictures of the same person 
    should give roughly the same measurements.

    Machine learning people call the 128 measurements of each face an embedding. The idea of reducing complicated raw data like a picture
    into a list of computer-generated numbers comes up a lot in machine learning 


    So all we need to do ourselves is run our face images through their pre-trained network to get the 128
    measurements for each face


    Run the openface scripts from inside the openface root directory:

First, do pose detection and alignment:

./util/align-dlib.py ./training-images/ align outerEyesAndNose ./aligned-images/ --size 96

Second, generate the representations from the aligned images:

./batch-represent/main.lua -outDir ./generated-embeddings/ -data ./aligned-images/

When you are done, the ./generated-embeddings/ folder will contain a csv file with the embeddings for each image.
'''

