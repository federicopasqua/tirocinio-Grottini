import argparse
import json

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('-gt', '--groundtruth', required=True, help='File path of coco ground truths')
parser.add_argument('-dt', '--detection', required=True, help='FIle path of Results of detection')
args = parser.parse_args()
    

dt = {}
with open(args.detection) as json_file:
    dt = json.load(json_file)

cocoGt=COCO(args.groundtruth)
cocoDt=cocoGt.loadRes(dt["results"])
E = COCOeval(cocoGt,cocoDt, 'bbox')
E.params.imgIds = dt["images"]
E.evaluate()
E.accumulate()
E.summarize()
