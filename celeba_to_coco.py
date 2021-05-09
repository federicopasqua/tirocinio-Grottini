import os
import json
import argparse

from PIL import Image

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-d', '--dataset', required=True, help='File path of directory containing the images')
parser.add_argument('-a', '--annotations', required=True, help='File path of annotations')
parser.add_argument('-o', '--output', required=True, help='File path of the annotations in coco format')
args = parser.parse_args()

coco = {
    "info" : {
        "year": 2021,
        "version": "1",
        "description": "description",
        "contributor": "me",
        "url": "url",
        "date_created": "2021/04/29",
    },
    "licenses": [{
                "id": 1,
                "name": "A license",
                "url": "url",
    }],
    "images": [],
    "annotations": [],
    "categories": [
        {
            "id": 0,
            "name": "face",
            "supercategory": "face" 
        }
    ]
}

images = []

for filename in os.listdir(args.dataset):
    if filename.endswith(".jpg"):
        path = args.dataset + filename if args.dataset.endswith("/") else args.dataset + "/" + filename
        try:
            image = Image.open(path)
        except:
            print("Could not open " + filename + ". Skipped...")
            continue
        width, height = image.size
        image = {
            "id": int(filename.split('.')[0]),
            "width": width,
            "height": height,
            "file_name": filename,
            "license": 1,
            "flickr_url": "url",
            "coco_url": "url",
            "date_captured": "2021/04/29",
        }
        coco["images"].append(image)
        images.append(int(filename.split('.')[0]))

file1 = open(args.annotations, 'r')
lines = file1.readlines()
i = -1
for line in lines:
    data = line.split(' ')
    data = [x for x in data if x]
    id = data[0].split('.')
    if len(id) == 2 and int(id[0]) in images:
        i += 1
        x = int(data[1])
        y = int(data[2])
        width = int(data[3])
        height = int(data[4][:-1])
        annotation = {
            "id": i,
            "image_id": int(id[0]),
            "category_id": 0,
            "area": width*height,
            "bbox": [x, y, width, height],
            "iscrowd": 0,
        }
        coco["annotations"].append(annotation)


with open(args.output, 'w') as outfile:
    json.dump(coco, outfile)










