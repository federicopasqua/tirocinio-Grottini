import argparse
import cv2
import time

import pandas as pd
import mediapipe as mp
mp_face_detection = mp.solutions.face_detection 


def postprocess_output(img_width, img_height, results):

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


        df = pd.DataFrame(boxes)
        df['ymin'] = df[0].apply(lambda y: max(1,(y*img_height)))
        df['xmin'] = df[1].apply(lambda x: max(1,(x*img_width)))
        df['ymax'] = df[2].apply(lambda y: min(img_height,(y*img_height)))
        df['xmax'] = df[3].apply(lambda x: min(img_width,(x * img_width)))
        boxes_scaled = df[['ymin', 'xmin', 'ymax', 'xmax']].values
    return boxes_scaled, classes, scores, num_det


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--camera_idx', type=int, help='Index of which video source to use. ', default = 0)
    parser.add_argument('-t', '--threshold', type=float, default=0.5, help='classifier score threshold')
    args = parser.parse_args()

    
    i = 0
    cap = cv2.VideoCapture('/dev/video0', cv2.CAP_V4L)
    start = time.time()
    while cap.isOpened() and i < 2000:
        ret, frame = cap.read()
        if not ret:
            break
        cv2_im = frame
        cv2_im_rgb = cv2.cvtColor(cv2_im, cv2.COLOR_BGR2RGB)
        height, width, channels = cv2_im_rgb.shape
        with mp_face_detection.FaceDetection(min_detection_confidence=args.threshold) as face_detection: 
            results = face_detection.process(cv2.cvtColor(cv2_im_rgb, cv2.COLOR_BGR2RGB))
            bb, classes, scores, num = postprocess_output(width, height, results)
            i += 1
            #cv2_im = append_objs_to_img(cv2_im, inference_size, objs, labels)
            for n in range(len(bb)):
                print(num, bb[n], scores[n])
        

    total_time = time.time() - start
    print("TOTAL TIME: " + str(total_time) + " ms")
    print("FPS: " + str(i / total_time))

    cap.release()
    cv2.destroyAllWindows()

def append_objs_to_img(cv2_im, inference_size, objs):
    height, width, channels = cv2_im.shape
    scale_x, scale_y = width / inference_size[0], height / inference_size[1]
    for obj in objs:
        bbox = obj.bbox.scale(scale_x, scale_y)
        x0, y0 = int(bbox.xmin), int(bbox.ymin)
        x1, y1 = int(bbox.xmax), int(bbox.ymax)

        percent = int(100 * obj.score)
        label = '{}%'.format(percent)

        cv2_im = cv2.rectangle(cv2_im, (x0, y0), (x1, y1), (0, 255, 0), 2)
        cv2_im = cv2.putText(cv2_im, label, (x0, y0+30),
                             cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
    return cv2_im

if __name__ == '__main__':
    main()