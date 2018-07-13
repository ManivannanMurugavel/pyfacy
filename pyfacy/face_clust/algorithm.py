from utils import load_images_to_clust_encodings,save_faces,selecting_clust_alg_model
import os

class Face_Clust_Algorithm():

	def __init__(self,dataset_path):
		self.valid_path = False
		if os.path.isdir(dataset_path):
			self.valid_path = True
			self.path = dataset_path
		else:
			print('[ERROR] Please set valid path')

	def load_faces(self):
		if self.valid_path:
			self.encs,self.paths = load_images_to_clust_encodings(self.path)
			self.model = selecting_clust_alg_model()
			self.model.fit(self.encs)
			self.tot_faces_list = self.model.labels_
		else:
			print('[ERROR] Please set Valid Path')

	def save_faces(self,save_location=None):
		if save_location is not None:
			save_faces(self.tot_faces_list,self.paths,save_location)
		else:
			save_faces(self.tot_faces_list,self.paths)
