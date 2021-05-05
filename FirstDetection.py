from imageai.Detection import ObjectDetection
import os

execution_path = os.getcwd()

detector = ObjectDetection()
detector.setModelTypeAsRetinaNet()
detector.setModelPath( os.path.join(execution_path , "/home/ryan/projects/vehicle-recognition/resnet50_coco_best_v2.1.0.h5"))
detector.loadModel()
detections = detector.detectObjectsFromImage(input_image=os.path.join(execution_path , "/home/ryan/projects/vehicle-recognition/Input/car.jpg"), output_image_path=os.path.join(execution_path , "/home/ryan/projects/vehicle-recognition/Output/newcar.jpg"))

for eachObject in detections:
    print(eachObject["name"] , " : " , eachObject["percentage_probability"] )
