from ultralytics import YOLO

def start_clinical_optimization(yaml_file, project_root):
    model = YOLO('yolov8n.pt')

    model.train(
        data=yaml_file,
        epochs=165,
        imgsz=800,
        batch=16,
        lr0=0.01,
        weight_decay=0.001,
        dropout=0.15,
        patience=20,
        name='skin_cancer_v4_80plus',
        project=f"{project_root}/runs",
        exist_ok=True,
        plots=True
    )