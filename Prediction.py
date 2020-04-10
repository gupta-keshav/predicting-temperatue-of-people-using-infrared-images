import cv2
import numpy as np
from yolo.yolo_model import YOLO
from argparse import ArgumentParser

def main(args):
	img = cv2.imread(args.imagePath)
	img = cv2.resize(img, (512, 512))
	objectName = args.objectName

	output = YOLO.predict(img, objectName, args.maxTemp, args.minTemp)


	cv2.imshow("Image", output)
	cv2.waitKey(10000)
	cv2.destroyAllWindows()

if __name__ == '__main__':
	parser = ArgumentParser()
	parser.add_argument('--imagePath', default='test.jpg')
	parser.add_argument('--objectName', default='person')
	parser.add_argument('--maxTemp', type=int, default=105)
	parser.add_argument('--minTemp', type=int, default=80)
	args = parser.parse_args()
	main(args)