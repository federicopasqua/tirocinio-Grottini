import argparse
import time
import os
import json

import numpy as np
from PIL import Image
#import tensorflow as tf
import pandas as pd
import cv2
import xml.etree.ElementTree as Et
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection 


def postprocess_output(image_path, results):
	""" Output post processing
	args:
		image_path: path to image
	returns:
		boxes: numpy array (num_det, 4) of boundary boxes at image scale
		classes: numpy array (num_det) of class index
		scores: numpy array (num_det) of scores
		num_det: (int) the number of detections
	"""

	boxes = []
	classes = []
	scores = []
	boxes_scaled = []
	num_det = 0
	if results.detections:
		for detection in results.detections:
			#print(detection)
			num_det += 1
			#print(detection.location_data.relative_bounding_box)
			scores.append(detection.score[0])
			bb = detection.location_data.relative_bounding_box
			boxes.append([bb.ymin, bb.xmin, bb.height + bb.ymin, bb.width + bb.xmin])
			classes.append(0)


		print(num_det, boxes, classes, scores)

		# Scale the output to the input image size
		img_width, img_height = Image.open(image_path).size # PIL


		df = pd.DataFrame(boxes)
		df['ymin'] = df[0].apply(lambda y: max(1,(y*img_height)))
		df['xmin'] = df[1].apply(lambda x: max(1,(x*img_width)))
		df['ymax'] = df[2].apply(lambda y: min(img_height,(y*img_height)))
		df['xmax'] = df[3].apply(lambda x: min(img_width,(x * img_width)))
		boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].values
	return boxes_scaled, classes, scores, num_det

def draw_boundaryboxes(boxes, path, name, dest_path, dt_scores):
	""" Draw the detection boundary boxes
	args:
		image_path: path to image
	"""
	# Draw detection boundary boxes
	image = cv2.imread(path + name)
	for i in range(len(boxes)):
		[ymin, xmin, ymax, xmax] = list(map(int, boxes[i]))
		cv2.rectangle(image, (xmin,ymin), (xmax,ymax), (10, 255, 0), 2)
		cv2.putText(image, '{}% score'.format(int(dt_scores[i]*100)), (xmin, ymin+20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (10,255,0), 1)

	cv2.imwrite(os.path.join(dest_path + name), image)
	#print("Saved at", saved_path)

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-i', '--input', required=True, help='File path to the directory containing the images')
	parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Score threshold for detected objects')
	parser.add_argument('-o', '--output',  help='File path for the result image with annotations')
	parser.add_argument('-co', '--cocooutput', help='File path for the result in COCO format')
	parser.add_argument('-v', '--verbose', action='store_true', help='Show output')
	parser.add_argument('-z', '--limit', type=int, help='maximum number of inference')

	args = parser.parse_args()

	input = args.input
	if (not input.endswith("/")):
		input += "/"

	
	total_inference_time = 0
	number_of_inferences = 0
	cocoarray = []
	images = []
	for filename in os.listdir(input):
		if (args.limit and number_of_inferences >= args.limit):
			break
		if filename.endswith(".jpg"):
			image = cv2.imread(input + filename) 
			print(filename)
			with mp_face_detection.FaceDetection(min_detection_confidence=args.threshold) as face_detection: 
				start_time = time.time()
				results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
				stop_time = time.time()
				if args.verbose:
					print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
				number_of_inferences += 1
				total_inference_time += (stop_time - start_time) * 1000
				boxes, classes, scores, num_det = postprocess_output(args.input + filename, results)
				if (args.output):
					draw_boundaryboxes(boxes, args.input, filename, args.output, scores)
				if args.cocooutput:
					id = int(filename.split(".")[0])
					for i in range(len(boxes)):
						[ymin, xmin, ymax, xmax] = list(map(int, boxes[i]))
						cocoarray.append({"image_id": id, "category_id": 0, "bbox": [int(xmin), int(ymin), int(xmax - xmin), int(ymax - ymin)], "score": float(scores[i])})
						images.append(id)

	print('-------RESULTS--------')
	print('IMAGES PROCESSED: %.2f ms' % (number_of_inferences))
	print('TOTAL INFERENCE TIME: %.2f ms' % (total_inference_time))
	print('AVERAGE INFERENCE TIME: %.2f ms' % (total_inference_time / number_of_inferences))

	if args.cocooutput:
		with open(args.cocooutput, 'w') as outfile:
			json.dump({"results": cocoarray, "images": images}, outfile)
