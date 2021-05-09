import argparse
import time
import os
import json

from PIL import Image
from PIL import ImageDraw

from pycoral.adapters import common
from pycoral.adapters import detect
from pycoral.utils.dataset import read_label_file
from pycoral.utils.edgetpu import make_interpreter


def draw_objects(draw, objs, labels):
  """Draws the bounding box and label for each object."""
  for obj in objs:
    bbox = obj.bbox
    draw.rectangle([(bbox.xmin, bbox.ymin), (bbox.xmax, bbox.ymax)], outline='red')
    draw.text((bbox.xmin + 10, bbox.ymin + 10), '%s\n%.2f' % (labels.get(obj.id, obj.id), obj.score), fill='red')


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument('-m', '--model', required=True, help='File path of .tflite file')
  parser.add_argument('-i', '--input', required=True, help='File path to the directory containing the images')
  parser.add_argument('-l', '--labels', help='File path of labels file')
  parser.add_argument('-t', '--threshold', type=float, default=0.4, help='Score threshold for detected objects')
  parser.add_argument('-o', '--output',  help='File path for the result image with annotations')
  parser.add_argument('-co', '--cocooutput', help='File path for the result in COCO format')
  parser.add_argument('-v', '--verbose', action='store_true', help='Show output')
  parser.add_argument('-z', '--limit', type=int, help='maximum number of inference')
  args = parser.parse_args()

  labels = read_label_file(args.labels) if args.labels else {}
  interpreter = make_interpreter(args.model)
  interpreter.allocate_tensors()

  input = args.input
  if (not input.endswith("/")):
    input += "/"

  print("Starting inference")

  total_inference_time = 0
  number_of_inferences = -1
  cocoarray = []
  images = []

  for filename in os.listdir(input):
    if args.limit and args.limit <= number_of_inferences:
      break
    if filename.endswith(".jpg"):
      image = Image.open(input + filename)
      _, scale = common.set_resized_input(interpreter, image.size, lambda size: image.resize(size, Image.ANTIALIAS))
      start = time.perf_counter()
      interpreter.invoke()
      inference_time = time.perf_counter() - start
      """Remove the first inference that takes additional time"""
      if (number_of_inferences != -1):
        total_inference_time += inference_time
      number_of_inferences += 1
      objs = detect.get_objects(interpreter, args.threshold, scale)
      

      if (args.verbose):
        print('%.2f ms' % (inference_time * 1000))
        if not objs:
          print('No objects detected')

      for obj in objs:
        if (args.verbose):
          print(labels.get(obj.id, obj.id))
          print('  id:    ', obj.id)
          print('  score: ', obj.score)
          print('  bbox:  ', obj.bbox)
        if args.cocooutput:
          id = int(filename.split(".")[0])
          cocoarray.append({"image_id": id, "category_id": 0, "bbox": [obj.bbox.xmin, obj.bbox.ymin, obj.bbox.xmax - obj.bbox.xmin, obj.bbox.ymax - obj.bbox.ymin], "score": obj.score})
          images.append(id)
      if args.output:
        image = image.convert('RGB')
        draw_objects(ImageDraw.Draw(image), objs, labels)
        image.save(args.output)

      
  print('-------RESULTS--------')
  print('IMAGES PROCESSED: %.2f ms' % (number_of_inferences))
  print('TOTAL INFERENCE TIME: %.2f ms' % (total_inference_time * 1000))
  print('AVERAGE INFERENCE TIME: %.2f ms' % (total_inference_time * 1000 / number_of_inferences))

  if args.cocooutput:
    with open(args.cocooutput, 'w') as outfile:
      json.dump({"results": cocoarray, "images": images}, outfile)


  
  
    


if __name__ == '__main__':
  main()
