from pyfacy import face_recog,utils



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
img = utils.load_image('6.jpg')
#Predict your image
prediction = mdl.predict(img)
print(prediction)
#Predict from File
prediction = mdl.predict_from_file('6.jpg')
print(prediction)

# Load model and Predict
mdl.load_model('model.pkl')
img = utils.load_image('6.jpg')
prediction = mdl.predict(img)
print(prediction)
#Predict from File
prediction = mdl.predict_from_file('6.jpg')
print(prediction)