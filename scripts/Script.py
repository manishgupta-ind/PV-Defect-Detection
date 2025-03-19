import os

print("Installing required packages...")
os.system("pip install ultralytics opencv-python pandas torch")
print("Package installation completed.\n")

import cv2
import pandas as pd
import torch
from ultralytics import YOLO

def convert_bbox_format(row, img_width, img_height):
    x_center = (row["xmin"] + row["xmax"]) / (2 * img_width)
    y_center = (row["ymin"] + row["ymax"]) / (2 * img_height)
    width = (row["xmax"] - row["xmin"]) / img_width
    height = (row["ymax"] - row["ymin"]) / img_height
    return f"{row['class']} {x_center} {y_center} {width} {height}\n"

def prepare_yolo_dataset():
    print("Preparing YOLO dataset...")
    TRAIN_DATA_DIR = "./Train"
    YOLO_IMAGES_DIR = "./yolo_dataset/images"
    YOLO_LABELS_DIR = "./yolo_dataset/labels"
    DATA_YAML_PATH = "./yolo_dataset/data.yaml"
    os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)
    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
    CSV_FILE = os.path.join(TRAIN_DATA_DIR, "_annotations.csv")
    df = pd.read_csv(CSV_FILE)
    defect_mapping = {'hotspot': 0, 'diode': 1, 'no_defect': 2}
    df["class"] = df["class"].map(defect_mapping)
    yolo_labels = {}
    for _, row in df.iterrows():
        image_path = os.path.join(TRAIN_DATA_DIR, row["filename"])
        new_image_path = os.path.join(YOLO_IMAGES_DIR, row["filename"])
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        cv2.imwrite(new_image_path, img)
        label_entry = convert_bbox_format(row, img_width, img_height)
        if label_entry:
            yolo_labels.setdefault(row["filename"], []).append(label_entry)
    for filename, labels in yolo_labels.items():
        label_path = os.path.join(YOLO_LABELS_DIR, filename.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.writelines(labels)
    data_yaml = f"""
    train: /home/jovyan/projects/yolo_dataset/images
    val: /home/jovyan/projects/yolo_dataset/images
    nc: 2
    names: ['hotspot', 'diode']
    """
    with open(DATA_YAML_PATH, "w") as f:
        f.write(data_yaml)
    print("YOLO dataset preparation completed.\n")
    return DATA_YAML_PATH

def train_model(data_yaml_path, model_save_path="best.pt"):
    print("Starting model training...")
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml_path, epochs=50, imgsz=416)
    model.save(model_save_path)
    print("Model training completed. Model saved at:", model_save_path, "\n")
    return model_save_path

def convert_bbox_format(row, img_width, img_height):
    x_center = (row["xmin"] + row["xmax"]) / (2 * img_width)
    y_center = (row["ymin"] + row["ymax"]) / (2 * img_height)
    width = (row["xmax"] - row["xmin"]) / img_width
    height = (row["ymax"] - row["ymin"]) / img_height
    return f"{row['class']} {x_center} {y_center} {width} {height}\n"

def prepare_yolo_dataset():
    print("Preparing YOLO dataset...")
    TRAIN_DATA_DIR = "./Train"
    YOLO_IMAGES_DIR = "./yolo_dataset/images"
    YOLO_LABELS_DIR = "./yolo_dataset/labels"
    DATA_YAML_PATH = "./yolo_dataset/data.yaml"
    os.makedirs(YOLO_IMAGES_DIR, exist_ok=True)
    os.makedirs(YOLO_LABELS_DIR, exist_ok=True)
    CSV_FILE = os.path.join(TRAIN_DATA_DIR, "_annotations.csv")
    df = pd.read_csv(CSV_FILE)
    defect_mapping = {'hotspot': 0, 'diode': 1, 'no_defect': 2}
    df["class"] = df["class"].map(defect_mapping)
    yolo_labels = {}
    for _, row in df.iterrows():
        image_path = os.path.join(TRAIN_DATA_DIR, row["filename"])
        new_image_path = os.path.join(YOLO_IMAGES_DIR, row["filename"])
        img = cv2.imread(image_path)
        if img is None:
            continue
        img_height, img_width = img.shape[:2]
        cv2.imwrite(new_image_path, img)
        label_entry = convert_bbox_format(row, img_width, img_height)
        if label_entry:
            yolo_labels.setdefault(row["filename"], []).append(label_entry)
    for filename, labels in yolo_labels.items():
        label_path = os.path.join(YOLO_LABELS_DIR, filename.replace(".jpg", ".txt"))
        with open(label_path, "w") as f:
            f.writelines(labels)
    data_yaml = f"""
    train: /home/jovyan/projects/yolo_dataset/images
    val: /home/jovyan/projects/yolo_dataset/images
    nc: 2
    names: ['hotspot', 'diode']
    """
    with open(DATA_YAML_PATH, "w") as f:
        f.write(data_yaml)
    print("YOLO dataset preparation completed.\n")
    return DATA_YAML_PATH

def train_model(data_yaml_path, model_save_path="best.pt"):
    print("Starting model training...")
    model = YOLO("yolov8n.pt")
    model.train(data=data_yaml_path, epochs=50, imgsz=416)
    model.save(model_save_path)
    print("Model training completed. Model saved at:", model_save_path, "\n")
    return model_save_path

def generate_predictions(model_path, test_data_dir="./Test", output_file="manish_gupta.csv"):
    print("Generating predictions...")
    test_images = os.listdir(test_data_dir)
    submission_data = []
    finetuned_model = YOLO("./runs/detect/train/weights/best.pt")
    CLASS_NAMES = {0: "hotspot", 1: "diode", 2: "no_defect"}

    for img_name in test_images:
        img_path = os.path.join(test_data_dir, img_name)

        results = finetuned_model(img_path)  # Run YOLO on the image
        boxes = results[0].boxes.xyxy  # Extract bounding boxes
        class_ids = results[0].boxes.cls  # Extract class predictions

        if boxes.shape[0] == 0:
            detections =  [["no_defect", [-1, -1, -1, -1]]]  # No defect detected
        else:
            detections = []
            for i in range(boxes.shape[0]):  # Iterate over all detected objects
                bbox = boxes[i].tolist()  # Convert bounding box to list
                class_id = int(class_ids[i].item())  # Get class ID
                class_name = CLASS_NAMES.get(class_id, "Unknown")  # Get class name
                detections.append([class_name, bbox[:4]])  

        for each_result in detections:
            # print(each_result)
            cls = each_result[0]
            xmin, ymin, xmax, ymax = map(int, each_result[1])
            submission_data.append([img_name[:-4], cls, xmin, ymin, xmax, ymax])
    
    df_submission = pd.DataFrame(submission_data, columns=["filename", "class", "xmin", "ymin", "xmax", "ymax"])
    df_submission.to_csv(output_file, index=False)
    print(f"Prediction completed successfully and submission file saved as {output_file}")

if __name__ == "__main__":
    print("Starting script now...")
    data_yaml_path = prepare_yolo_dataset()
    model_path = train_model(data_yaml_path)
    generate_predictions(model_path)
    print("Script execution completed.")