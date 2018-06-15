import pyfacy
from pyfacy import utils

#Load the Images
known_image_1 = utils.load_image('./images/2.jpg')
known_image_2 = utils.load_image('./images/4.jpg')

#Find the Face Encodings for known images
known_image_1_encodings = utils.img_to_encodings(known_image_1)[0]
known_image_2_encodings = utils.img_to_encodings(known_image_2)[0]

encodings = [known_image_1_encodings,known_image_2_encodings]

# Load Test image and find encodings
test_image = utils.load_image('./images/11.jpg')
test_image_encodings = utils.img_to_encodings(test_image)[0]


#Distance Between Known images and Test Image
for known_image_enc in encodings:
	distance = utils.face_distance(known_image_enc,test_image_encodings)
	print(distance)

