from roboflow import Roboflow


rf = Roboflow(api_key="")

print(rf.workspace())

workspaceId = 'andriiiiiiiiiiii'
projectId = 'manhua-bubbles'
project = rf.workspace(workspaceId).project(projectId)

project.upload(
    image_path="E:/Projects/Python/ComicGlot/training/detection/data_labeling/raw_yolo_dataset",
    batch_name="Annotating",
    num_retry_uploads=3,
    tag_names=["autogen#1"],
)
                    