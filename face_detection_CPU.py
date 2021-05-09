import argparse
import time
import os
import json

import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import xml.etree.ElementTree as Et

def prepare_input(image_path):
	""" Input image preprocessing for SSD MobileNet format
	args:
		image_path: path to image
	returns:
		input_data: numpy array of shape (1, width, height, channel) after preprocessing
	"""
	# NxHxWxC, H:1, W:2
	height = input_details[0]['shape'][1]
	width = input_details[0]['shape'][2]
	img = Image.open(image_path).convert('RGB').resize((width, height))

	# add N dim
	input_data = np.expand_dims(img, axis=0)

	return input_data

def postprocess_output(image_path, threshold):
	""" Output post processing
	args:
		image_path: path to image
	returns:
		boxes: numpy array (num_det, 4) of boundary boxes at image scale
		classes: numpy array (num_det) of class index
		scores: numpy array (num_det) of scores
		num_det: (int) the number of detections
	"""
	# SSD Mobilenet tflite model returns 10 boxes by default.
	# Use the output tensor at 4th index to get the number of valid boxes
	num_det = int(interpreter.get_tensor(output_details[3]['index']))
	unfiltered_boxes = interpreter.get_tensor(output_details[0]['index'])[0][:num_det]
	unfiltered_classes = interpreter.get_tensor(output_details[1]['index'])[0][:num_det]
	unfiltered_scores = interpreter.get_tensor(output_details[2]['index'])[0][:num_det]

	filter_array = []

	for i in range(num_det):
		if (unfiltered_scores[i] < threshold):
			filter_array.append(False)
		else:
			filter_array.append(True)


	boxes = unfiltered_boxes[filter_array]
	classes = unfiltered_classes[filter_array]
	scores = unfiltered_scores[filter_array]
	num_det = len(scores)


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

def draw_boundaryboxes(boxes, path, name, dest_path):
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
	parser.add_argument('-m', '--model', required=True, help='File path of .tflite file')
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

	interpreter = tf.lite.Interpreter(model_path=args.model)
	interpreter.allocate_tensors()

	input_details = interpreter.get_input_details()
	output_details = interpreter.get_output_details()
	total_inference_time = 0
	number_of_inferences = 0
	cocoarray = []
	images = []
	for filename in os.listdir(input):
		if (args.limit and number_of_inferences >= args.limit):
			break
		if filename.endswith(".jpg"):
			input_data = prepare_input(input + filename)
			interpreter.set_tensor(input_details[0]['index'], input_data)

			start_time = time.time()
			interpreter.invoke()
			stop_time = time.time()
			if args.verbose:
				print('time: {:.3f}ms'.format((stop_time - start_time) * 1000))
			number_of_inferences += 1
			total_inference_time += (stop_time - start_time) * 1000
			boxes, classes, scores, num_det = postprocess_output(args.input + filename, args.threshold)
			if (args.output):
				draw_boundaryboxes(boxes, args.input, filename, args.output)
			if args.cocooutput:
				id = int(filename.split(".")[0])
				for i in range(len(boxes)):
					[ymin, xmin, ymax, xmax] = list(map(int, boxes[i]))
					cocoarray.append({"image_id": id, "category_id": 0, "bbox": [float(xmin), float(ymin), float(xmax - xmin), float(ymax - ymin)], "score": float(scores[i])})
					images.append(id)

	print('-------RESULTS--------')
	print('IMAGES PROCESSED: %.2f ms' % (number_of_inferences))
	print('TOTAL INFERENCE TIME: %.2f ms' % (total_inference_time))
	print('AVERAGE INFERENCE TIME: %.2f ms' % (total_inference_time / number_of_inferences))

	if args.cocooutput:
		with open(args.cocooutput, 'w') as outfile:
			json.dump({"results": cocoarray, "images": images}, outfile)
