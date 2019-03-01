# This Package used for Face Recognition with Machine Algorithm

## Installing Steps for requirements python package
### Installing dlib on Ubuntu
The following instructions were gathered on Ubuntu 16.04 but should work on newer versions of Ubuntu as well.

To get started, let’s install our required dependencies:

```
sudo apt-get update
sudo apt-get install build-essential cmake
sudo apt-get install libopenblas-dev liblapack-dev
sudo apt-get install libx11-dev libgtk-3-dev
sudo apt-get install python python-dev python-pip
sudo apt-get install python3 python3-dev python3-pip
```
after
```
pip install dlib
```
### Installing pyfacy models on Ubuntu

```
pip install pyfacy_dlib_models
```
### Installing imutils on Ubuntu
```
pip install imutils
```
### Installing numpy, scipy and sklearn
```
pip install numpy
pip install scipy
pip install scikit-learn
```
### Installing pyfacy
```
pip install pyfacy
```

## It's implemented with face encodings

## Examples:

#### Read Image
```
from pyfacy import utils
img = utils.load_image('<image src>')
ex:
img = utils.load_image('manivannan.jpg')
```

### Face Encodings:
```
from pyfacy import utils
img = utils.load_image('<image src>')
encodings = utils.img_to_encodings(img)
```

### Compare Two faces
```
from pyfacy import utils
image1 = utils.load_image('<image1 src>')
image2 = utils.load_image('<image2 src>')
matching,distance = utils.compare_faces(image1,image2)
```
> Note: The compare_faces return Boolean and Distance between two faces

# Example for Face Recognition using ML

## Implementing Algorithms

1. KNN - K-Nearest Neighbors
2. LOG_REG_BIN - Logistic Regression with two classes
3. LOG_REG_MUL - Logistic Regression with more than two classes
4. LDA - Linear Discriminant Analysis

### Training , Save model and Predict Image
```
from pyfacy import face_recog
from pyfacy import utils
mdl = face_recog.Face_Recog_Algorithm()
# Train the Model
# Implemented only four algorithms above mentioned and put the shortform
mdl.train('pyfacy/Test_DS',alg='LOG_REG_MUL')
# Save the Model
mdl.save_model()
# Predicting Image
img = utils.load_image('<image src>')
mdl.predict(img)
```

### Loading model and Predict Image
```
from pyfacy import face_recog
from pyfacy import utils
mdl = face_recog.Face_Recog_Algorithm()
# Load Model
mdl.load_model('model.pkl')
# Predicting Image
img = utils.load_image('<image src>')
mdl.predict(img)
```

# Face Clustering
### Cluster the image_src
```
from pyfacy import face_clust
# Create object for Cluster class with your source path(only contains jpg images)
mdl = face_clust.Face_Clust_Algorithm('./pyfacy/cluster')
# Load the faces to the algorithm
mdl.load_faces()
# Save the group of images to custom location(if the arg is empty store to current location)
mdl.save_faces('./pyfacy')
