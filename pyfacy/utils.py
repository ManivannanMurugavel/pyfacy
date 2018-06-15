import dlib
from scipy import misc
import numpy as np
from imutils import paths
import os
import random
from pyfacy_dlib_models import dlib_face_recognition,shape_predictor_5
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

pose_predictor_5_point_url = shape_predictor_5()

face_encoder_url = dlib_face_recognition()

face_detector = dlib.get_frontal_face_detector()


pose_predictor_5_point = dlib.shape_predictor(pose_predictor_5_point_url)

face_encoder = dlib.face_recognition_model_v1(face_encoder_url)

def load_image(image_src):
	img = misc.imread(image_src)
	return img

def dlib_rect_to_css(rect):
	#:return: a plain tuple representation of the rect in (top, right, bottom, left) order
	return (rect.top(), rect.right(), rect.bottom(), rect.left())

def css_to_dlib_rect(css):
	#Convert a tuple in (top, right, bottom, left) order to a dlib `rect` object
	return dlib.rectangle(css[3], css[0], css[1], css[2])
	


def detect_faces_locations_from_dlib(img,number_of_times_to_upsample=1, model="cpu"):
		return face_detector(img, number_of_times_to_upsample)

def detect_faces_locations_from_opencv(img):
	return "Feature Enhancement"

def detect_one_face_axies(face_location):
	return (face_location.top(),face_location.right(),face_location.bottom(),face_location.left())

def detect_faces_axies_from_dlib(dlib_face_locations):
	face_axies = {}
	face = {}
	if not isinstance(dlib_face_locations,dlib.rectangles):
		print('[INFO] Send correct dlib_face_locations')
		return 0
	dlib_face_locations_length = len(dlib_face_locations)
	face_axies['tot_faces'] = dlib_face_locations_length
	for num_face,dlib_face_location in zip(range(dlib_face_locations_length),dlib_face_locations):
		face['top'] = dlib_face_location.top()
		face['left'] = dlib_face_location.left()
		face['bottom'] = dlib_face_location.bottom()
		face['right'] = dlib_face_location.right()
		face_axies['face_'+str(num_face+1)] = face
	return face_axies

def detect_faces_axies_from_opencv(img):
	return "Feature Enhancement"

def detect_faces_axies(dlib_face_locations,lib='dlib'):
	if lib == 'dlib':
		return detect_faces_axies_from_dlib(dlib_face_locations)
	elif lib == 'opencv':
		return detect_faces_axies_from_opencv()


def detect_faces(img,lib='dlib'):
	if lib == 'dlib':
		return detect_faces_locations_from_dlib(img)
	elif lib == 'opencv':
		return detect_faces_locations_from_opencv(img)


def detect_face_landmarks(face_image,face_locations=None, lib="dlib", model="large"):
	if face_locations is None:
		face_locations = detect_faces(face_image,lib)
	
	# pose_predictor = pose_predictor_68_point

	if model == "small":
		pose_predictor = pose_predictor_5_point
	return [pose_predictor(face_image, face_location) for face_location in face_locations]

def img_to_encodings(face_image,known_face_locations=None, num_jitters=1,lib="dlib"):
	"""
	:param img: The image that contains one or more faces
	:param known_face_locations: Optional - the bounding boxes of each face if you already know them.
	:param num_jitters: How many times to re-sample the face when calculating encoding. Higher is more accurate, but slower (i.e. 100 is 100x slower)
    :return: A list of 128-dimensional face encodings (one for each face in the image)
	"""
	raw_landmarks = detect_face_landmarks(face_image, known_face_locations, lib, model="small")

	return [np.array(face_encoder.compute_face_descriptor(face_image, raw_landmark_set, num_jitters)) for raw_landmark_set in raw_landmarks]

def load_images_to_encodings(face_paths):
	imagePaths = sorted(list(paths.list_images(face_paths)))
	random.seed(42)
	random.shuffle(imagePaths)
	tot_len = len(imagePaths)
	print("Total Images is {}".format(tot_len))
	X = []
	y = []
	print("[INFO] Reading face from images")
	for idx,imagePath in enumerate(imagePaths):
		if imagePath.endswith('.jpg'):
			print("{} / {}".format(idx+1,tot_len))
			face = img_to_encodings(misc.imread(imagePath))
			if len(face) == 1:
				X.append(face[0])
				y.append(imagePath.split(os.path.sep)[-2])
			else:
				print("[INFO] This {} image did't have face \nor have more then one face".format(imagePath))
		else:
			print("[INFO] This Package only support .jpg files")
	unique_names = list(set(y))
	labels = [unique_names.index(name) for name in y]
	return (np.array(X),np.array(labels),unique_names,tot_len)

def selecting_alg_model(alg,records,un_cnt):
	alg = alg.upper()
	if un_cnt < 2:
		print("[ERROR] Please use more then 1 faces")
		exit()
	if un_cnt == 2 and alg != 'LOG_REG_BIN':
		print("[WARNINGS] You can choose LOG_REG_BIN algorithm for Binary Classification")
	if alg == 'KNN':
		if records <= 3:
			print("[INFO] Please use more then 3 records to KNN")
			return 0
		return KNeighborsClassifier(n_neighbors=3)
	elif alg == 'LOG_REG_MUL':
		return LogisticRegression(class_weight='balanced',solver='newton-cg',multi_class='multinomial')
	elif alg == 'LOG_REG_BIN':
		return LogisticRegression(class_weight='balanced',multi_class='ovr')
	elif alg == 'LDA':
		return LinearDiscriminantAnalysis()
	else:
		print("[INFO] Please select correct Algorithm Name")

def face_distance(known_face_encodings, unknown_face_encodings):
	return np.linalg.norm(known_face_encodings - unknown_face_encodings)

def compare_faces(known_face_src,unknown_face_src,diff=0.5):
	known_face_encodings = img_to_encodings(misc.imread(known_face_src))[0]
	unknown_face_encodings = img_to_encodings(misc.imread(unknown_face_src))[0]
	prob = face_distance(known_face_encodings, unknown_face_encodings)
	if prob <= diff:
		return (True,prob)
	else:
		return (False,prob)