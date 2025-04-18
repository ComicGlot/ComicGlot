import roboflow
from roboflow import Roboflow

rf = roboflow.Roboflow(api_key="***REMOVED***")
project = rf.workspace().project("manhua-bubbles")
 
version = project.version(6)
dataset = version.download("yolov11")

version.deploy("yolov11","runs/detect/train16")
                