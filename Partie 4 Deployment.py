from flask import Flask, render_template, request
import cv2
import numpy as np
import matplotlib.pyplot as plt
from copy import deepcopy
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import ZeroPadding2D, Convolution2D, MaxPooling2D, Dropout, Flatten, Activation
from keras.models import model_from_json
from tensorflow.keras.utils import img_to_array
from tensorflow.keras.utils import load_img
from keras import Model
import numpy as np
from keras.applications.imagenet_utils import preprocess_input
import os
import cv2

def card_cropping(IMAGE_PATH):
    refFilename= "zyro-image.png"
    im1 = cv2.imread(refFilename, cv2.IMREAD_COLOR)
    im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)

    imFilename= IMAGE_PATH
    im2 = cv2.imread(imFilename, cv2.IMREAD_COLOR)
    im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
    
    im1_gray = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    im2_gray = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    MAX_NUM_FEATURES = 1000
    orb = cv2.ORB_create(MAX_NUM_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(im1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(im2_gray, None)

    im1_display = cv2.drawKeypoints(im1, keypoints1, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    im2_display = cv2.drawKeypoints(im2, keypoints2, outImage=np.array([]), color=(255, 0, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
    matcher = cv2.DescriptorMatcher_create(cv2.DESCRIPTOR_MATCHER_BRUTEFORCE_HAMMING)
    matches = matcher.match(descriptors1, descriptors2, None)

    matches = sorted(matches, key = lambda x:x.distance)

    numGoodMatches = int(len(matches) * 0.1)
    matches = matches[:numGoodMatches]

    im_matches = cv2.drawMatches(im1, keypoints1, im2, keypoints2, matches, None)
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
      points1[i, :] = keypoints1[match.queryIdx].pt
      points2[i, :] = keypoints2[match.trainIdx].pt

    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)
    height, width, channels = im1.shape
    im2_reg = cv2.warpPerspective(im2, h, (width, height))
    
    im2_reg_fix = cv2.cvtColor(im2_reg, cv2.COLOR_BGR2RGB)
    cv2.imwrite("cropped student card.png", im2_reg_fix)
    
def face_cropping():
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

    img = cv2.imread('cropped student card.png')
    cropped_face = deepcopy(img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.1, 4)

    a,c,b,d=0,0,0,0

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 10)
        (a,c,b,d)=(x,x+w,y,h+y)
        print(x,w,y,h)

    cropped_face = cropped_face[b:d , a:c]
    
    img_fix = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    cv2.imwrite("cropped face.png", cropped_face)
    
def vgg_face():	
    model = Sequential()
    model.add(ZeroPadding2D((1,1),input_shape=(224,224, 3)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(128, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
        
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(256, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(ZeroPadding2D((1,1)))
    model.add(Convolution2D(512, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2,2), strides=(2,2)))
    
    model.add(Convolution2D(4096, (7, 7), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(4096, (1, 1), activation='relu'))
    model.add(Dropout(0.5))
    model.add(Convolution2D(2622, (1, 1)))
    model.add(Flatten())
    model.add(Activation('softmax'))
    return model

def load_images_from_folder(folder):
    image_paths = {}
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            label = os.path.splitext(filename)[0]
            image_paths[label] = img_path
    return image_paths

def preprocess_image(image_path):
    img = load_img(image_path, target_size=(224, 224))
    img = img_to_array(img)
    img = np.expand_dims(img, axis=0)
    img = preprocess_input(img)
    return img

def findCosineSimilarity(source_representation, test_representation):
    a = np.matmul(np.transpose(source_representation), test_representation)
    b = np.sum(np.multiply(source_representation, source_representation))
    c = np.sum(np.multiply(test_representation, test_representation))
    return 1 - (a / (np.sqrt(b) * np.sqrt(c)))
        
        
def final(IMAGE_PATH):
    card_cropping(IMAGE_PATH)
    face_cropping()

app= Flask(__name__)

@app.route('/', methods=['GET'])
def hello_world():
    return render_template('First page.html')

@app.route('/', methods=['POST'])
def predict():
    imagefile = request.files['imagefile']
    IMAGE_PATH = "./images/" + imagefile.filename
    imagefile.save(IMAGE_PATH)
    
    final(IMAGE_PATH)
    model = vgg_face()
    model.load_weights('vgg_face_weights.h5')

    vgg_face_descriptor = Model(inputs=model.layers[0].input, outputs=model.layers[-2].output)
    
    folder = "persons"
    student_images_dict = load_images_from_folder(folder)
    
    names = list(student_images_dict.keys())
    paths = list(student_images_dict.values())

    input_image = vgg_face_descriptor.predict(preprocess_image('cropped face.png'))[0,:]
    epsilon = 0.40
    verified=[]

    for i in range(0,len(paths)):
        data_image=vgg_face_descriptor.predict(preprocess_image(paths[i]))[0,:]

        cosine_similarity = findCosineSimilarity(input_image, data_image)

        if(cosine_similarity < epsilon):
            print("verified... they are same person", cosine_similarity)
            verified.append(i)
        else:
            print("unverified! they are not same person!", cosine_similarity)

    if len(verified)==0:
        name= "This person is not in the database"
    elif len(verified)==1:
        name= "This person is: "+names[verified[0]]
    else:
        name= "There is an error, the model corresponded this face to more than one person"
        print(verified)
    
    return render_template('First page.html', prediction=name)
    
if __name__ == "__main__":
    app.run(port=80, debug=False)
