import cv2
import numpy as np
import os

class YOLO:
	def predict(img, objectName, maxTemp, minTemp):
		_net_weights_path = "C:/Users/Keshav Gupta/Desktop/yolo/Project/yolo/yolov3.weights"
		_net_cfg_path = "C:/Users/Keshav Gupta/Desktop/yolo/Project/yolo/yolov3.cfg"
		_net_names_path = "C:/Users/Keshav Gupta/Desktop/yolo/Project/yolo/coco.names"
		person_count = 1
		net = cv2.dnn.readNet(_net_weights_path, _net_cfg_path)
		classes = []
		with open(_net_names_path, "r") as f:
			classes = [line.strip() for line in f.readlines()]

		layer_names = net.getLayerNames()
		output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

		height, width, channels = img.shape

		blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)

		net.setInput(blob)
		outs = net.forward(output_layers)

		class_ids = []
		confidences = []
		boxes = []
		for out in outs:
			for detection in out:
				score = detection[5:]
				class_id = np.argmax(score)
				confidence = score[class_id]
				if confidence > 0.3:
					center_x = int(detection[0] * width)
					center_y = int(detection[1] * height)
					w = int(detection[2] * width)
					h = int(detection[3] * height)

					x = int(center_x - w/2)
					y = int(center_y - h/2)

					boxes.append([x, y, w, h])
					confidences.append(float(confidence))
					class_ids.append(class_id)		

		indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
		font = cv2.FONT_HERSHEY_PLAIN
		img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
		for i in range(len(boxes)):
			if i in indexes:
				x, y, w, h = boxes[i]
				# x += 10
				# y += 10
				# w -= 10
				# h -= 10
				label = str(classes[class_ids[i]])
				if label == objectName:
					cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
					max_intensity = 0
					for j in range(y+1,y+h):
						for k in range(x+1,x+w):
							box = img_gray[y-1:y, x-1:x]
							intensity_box = np.mean(box)
							if intensity_box > max_intensity:
								max_intensity = intensity_box
					temp = max_intensity * ((maxTemp - minTemp)/255) + minTemp
					text = str(person_count) + ' ' + str(int(temp))	
					cv2.putText(img, str(text), (x, y-10), font, 1, (255, 255, 255), 2)
					person_count += 1


		return img