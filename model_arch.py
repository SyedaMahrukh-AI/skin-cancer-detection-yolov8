
from ultralytics import YOLO

def load_skin_model(model_path="models/best.pt"):

    return YOLO(model_path)