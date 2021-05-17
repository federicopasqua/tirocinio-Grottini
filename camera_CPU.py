import argparse
import cv2
import time

import numpy as np
from PIL import Image
import tensorflow as tf
import pandas as pd
import cv2
import xml.etree.ElementTree as Et

def postprocess_output(img_width, img_height, threshold, output_details, interpreter):
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


    df = pd.DataFrame(boxes)
    df['ymin'] = df[0].apply(lambda y: max(1,(y*img_height)))
    df['xmin'] = df[1].apply(lambda x: max(1,(x*img_width)))
    df['ymax'] = df[2].apply(lambda y: min(img_height,(y*img_height)))
    df['xmax'] = df[3].apply(lambda x: min(img_width,(x * img_width)))
    boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].values
    return boxes_scaled, classes, scores, num_det

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model', help='.tflite model path')
    parser.add_argument('-c', '--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='classifier score threshold')
    args = parser.parse_args()

    interpreter = tf.lite.Interpreter(model_path=args.model)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    height = input_details[0]['shape'][1]
    width = input_details[0]['shape'][2]
    i = 0
    cap = cv2.VideoCapture(args.camera_idx)
    start = time.time()
    while cap.isOpened() and i < 500:
        ret, frame = cap.read()
        if not ret:
            break
        #cv2_im = frame
        img = Image.fromarray(frame).convert('RGB').resize((width, height))
        input_data = np.expand_dims(img, axis=0)
        #cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        #cv2_im_rgb = cv2.resize(cv2_im_rgb, (height, width))
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        bb, classes, scores, num = postprocess_output(width, height, 0.4, output_details, interpreter)
        i += 1
        #cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
        for n in range(len(bb)):
            print(num, bb[n], scores[n])


        #cv2.imshow('frame', cv2_im)
        #if cv2.waitKey(1) & 0xFF == ord('q'):
        #    break

    total_time = time.time() - start
    print("TOTAL TIME: " + str(total_time) + " ms")
    print("FPS: " + str(i / total_time))

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs, labels):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}% {}'.format(percent, labels.get(obj.id, obj.id))

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()