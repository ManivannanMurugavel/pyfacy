# This Package used for Face Recognition with Machine Algorithm

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