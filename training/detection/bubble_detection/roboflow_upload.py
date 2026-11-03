import roboflow
from roboflow import Roboflow

import os
from dotenv import load_dotenv
load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY", "")

rf = roboflow.Roboflow(api_key=roboflow_api_key)
project = rf.workspace().project("manhua-bubbles")
 
version = project.version(6)
dataset = version.download("yolov11")

version.deploy("yolov11","runs/detect/train16")
                