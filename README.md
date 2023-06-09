<h1 align="center">Photos recognition on a student ID using Deep Learning</h1>

## Abstract
The objective of this project was to implement a photo recognition system on student ID cards, as well as to detect the text on these cards, and then compare the information on the studied student ID card with the information in our database to determine if they are compatible or not. This is aimed at reducing fraud related to student ID cards.
Given the great success of Convolutional Neural Networks (CNNs) in image classification and recognition, we used a deep learning method to build our model.
Firstly, we performed data preprocessing by resizing the student ID cards to reduce the complexity of the model and decrease the computing power required for training.
Next, we used pre-trained models to extract the information from the student ID cards. To automatically extract the text, we used keras-ocr. To automatically extract faces from the student ID cards, we used the pre-trained Haar Cascade Frontal Face model. We then processed and resized the face image to prepare it for our facial recognition model.
Finally, we applied our facial recognition model to the image and returned the person’s name.
We then deployed our model on a website to facilitate its use, using Flask as an API and HTML/CSS for the frontend
## Report & Presentation
You can read the report and see the presentation here:
* Report
* Presentation
## Requirements
You can download the VGG Face weight.h5 from here: "https://github.com/serengil/deepface_models/releases/download/v1.0/vgg_face_weights.h5"
