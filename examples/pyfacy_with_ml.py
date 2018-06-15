import pyfacy
from pyfacy import face_recog


#Create the object for Class
mdl = face_recog.Face_Recog_Algorithm()

# Train with Your Algorithms
'''
1. KNN
2. LOG_REG_BIN - two class
3. LOG_REG_MUL - more than two classes
4. LDA
'''
mdl.train('Dataset/',alg='LOG_REG_MUL')
# Save your model with model.pkl
mdl.save_model()
#Load you Image
img = utils.load_image('<image url>')
#Predict your image
mdl.predict(img)
#Predict from File
mdl.predict_from_file('sleep1025.jpg')


# Load model and Predict
mdl.load_model('model.pkl')
mdl.predict(img)
#Predict from File
mdl.predict_from_file('sleep1025.jpg')