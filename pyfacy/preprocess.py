import utils
import cv2





def imageRead(url,gray=False):
	img = cv2.imread(url)