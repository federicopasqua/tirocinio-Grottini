# tirocinio-grottini

### Inference CPU  
```
python3 face_detection_CPU.py --model ssd_mobilenet_v2_face_quant_postprocess.tflite --input ~/datasets/img_celeba/ -co ./results_coco_cpu -v --limit 2000  
```
### Inference coral  
```
python3 face_detection_coral.py --model ssd_mobilenet_v2_face_quant_postprocess_edgetpu.tflite --input ~/datasets/img_celeba/ -co ./results/results_coco_coral -v --limit 5000  
```
  
### convert celebA -> COCO 
```
python3 celeba_to_coco.py -d ~/datasets/img_celeba/ -a ./annotazioni_celeba -o ./coco_ann  
```
  
### Calcolo metriche COCO  
```
python3 coco_faces.py -gt coco_ann -dt ./results/results_coco_cpu  
python3 coco_faces.py -gt coco_ann -dt ./results/results_coco_coral
```
