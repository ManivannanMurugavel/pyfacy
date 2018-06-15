import pyfacy
from pyfacy import utils


def status(comp,dis):
	if comp:
		print("Two faces is same and distance is {}".format(dis))
	else:
		print("Two faces is not same and distance is {}".format(dis))

#Comparing two faces, it return True/False and distance
comp,dis = utils.compare_faces('./images/4.jpg', './images/11.jpg')

status(comp, dis)

comp,dis = utils.compare_faces('./images/4.jpg', './images/6.jpg')

status(comp, dis)