import os
import sys
import cv2
from ultralytics import YOLO

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

def main():
    model_path = os.path.join(BASE_DIR, "models", "best.pt")
    eval_folder = os.path.join(BASE_DIR, "EvalMedia")

    if not os.path.exists(model_path) or not os.path.exists(eval_folder):
        print("Required folders or model not found.")
        return

    try:
        model = YOLO(model_path)
        
        # Get list of all image files in EvalMedia
        valid_extensions = ('.jpg', '.jpeg', '.png')
        images = [f for f in os.listdir(eval_folder) if f.lower().endswith(valid_extensions)]

        if not images:
            print("No images found in EvalMedia.")
            return

        print(f"Testing {len(images)} images found in {eval_folder}")

        for image_name in images:
            image_path = os.path.join(eval_folder, image_name)
            results = model(image_path)

            for r in results:
                annotated_frame = r.plot()
                cv2.imshow(f"Detection: {image_name}", annotated_frame)
                
                print(f"Showing results for {image_name}. Press any key for next image.")
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            
    except Exception as e:
        print(f"Execution Error: {e}")

if __name__ == "__main__":
    main()