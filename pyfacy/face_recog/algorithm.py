import preprocess
from utils import img_to_encodings,load_images_to_encodings,selecting_alg_model
import random
import pickle
import os
import numpy as np
from imutils import paths
from scipy import misc
from sklearn.model_selection import train_test_split

class Face_Recog_Algorithm():
	def __init__(self,alg=None):
		self.alg = alg

	def train(self,src_path,test_size=0.2,alg=None):
		self.faces = []
		self.labels = []
		self.algorithms = ['KNN','LOG_REG_BIN','LOG_REG_MUL','LDA']
		
		if not os.path.isdir(os.path.join('./',src_path)):
			print("[INFO] Please select valid path")
			return 0
		if self.alg is None and alg is None:
			print("[INFO] Please Select any Algorithm from these ['KNN','Log_Reg_Bin','Log_Reg_Mul','LDA']")
			return 0
		elif alg is not None:
			self.alg = alg
		if self.alg.upper() not in self.algorithms:
			print("[INFO] Please Select valid Algorithm from these ['KNN','Log_Reg_Bin','Log_Reg_Mul','LDA']")
			return 0
		print("[INFO] loading images...")
		self.X,self.y,self.unique_names,self.tot_rec = load_images_to_encodings(src_path)
		print("[INFO] Selecting your {} Model".format(self.alg))
		self.model = selecting_alg_model(self.alg,self.tot_rec,len(self.unique_names))
		self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=test_size)
		print("[INFO] Fitting the Model")
		self.fitting_model()
		print("[INFO] Evaluting the Model")
		self.evaluate_model()
		return self.model
	def fitting_model(self):
		self.model.fit(self.X_train, self.y_train)
	def evaluate_model(self):
		score = self.model.score(self.X_test, self.y_test)
		print("[INFO] Evaluation Acc is {}%".format(score*100))
	def save_model(self):
		save_model = (self.model,self.unique_names)
		output = open('model.pkl', 'wb')
		pickle.dump(save_model, output)
		output.close()
		print("[INFO] Your Model save to current directory and the model name is model.pkl")
	def load_model(self,model):
		pkl_file = open('model.pkl', 'rb')
		pickle_data = pickle.load(pkl_file)
		pkl_file.close()
		self.model,self.unique_names = pickle_data

	def predict(self,img):
		# print(len(img_to_encodings(img)))
		prediction = []
		encodings = img_to_encodings(img)
		if len(encodings) <= 0:
			print("[INFO] Face is not identified.")
			exit()
		face_name = self.model.predict(encodings)
		prob = self.model.predict_proba(encodings)
		for name,pb in zip(face_name, prob):
			prediction.append((self.unique_names[name],round(pb[name]*100,3)))
		return prediction
		# print("{} with prob {}".format(self.unique_names[face_name],prob[np.argmax(prob)]*100))
	def predict_from_file(self,img_src):
		img = misc.imread(img_src)
		return self.predict(img)
