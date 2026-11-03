import roboflow
import os
from dotenv import load_dotenv
load_dotenv()
roboflow_api_key = os.getenv("ROBOFLOW_API_KEY")

rf = roboflow.Roboflow(api_key=roboflow_api_key)
project = rf.workspace().project("manhua-bubbles") 
print(rf.workspace())

project.upload(
    image_path="E:/Projects/Python/ComicGlot/training/detection/data_labeling/raw_yolo_dataset",
    batch_name="Annotating",
    num_retry_uploads=3,
    tag_names=["autogen#1"],
)
                    

# $ pip install roboflow
# $ roboflow login
# $ roboflow import -w andriiiiiiiiiiii -p manhua-bubbles raw_yolo_dataset
                    