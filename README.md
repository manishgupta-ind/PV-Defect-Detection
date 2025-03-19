# Defect Detection in Photovoltaic (PV) Modules

## Overview
This project focuses on detecting and classifying defects in photovoltaic (PV) modules using deep learning techniques. The problem involves both image classification and object detection, identifying defects such as **hotspot defects** and **diode failures** in thermal images. The model leverages the **YOLOv8** object detection framework for robust and efficient defect detection.

## Problem Statement
PV modules are prone to defects that impact their efficiency and reliability. Traditional manual inspections are slow and error-prone, necessitating an **automated computer vision-based solution**. The objective of this project is to develop an AI-ML model that can:

- **Classify** defects into three categories: **hotspot defect**, **diode failure**, or **no defect**.
- **Detect** and **localize** defective regions in PV module thermal images.

## Approach
### 1. Data Understanding & Preprocessing
- The dataset consists of thermal images with corresponding **bounding box annotations**.
- Conducted **Exploratory Data Analysis (EDA)**:
  - **Class distribution analysis** to check for imbalances.
  - **Image size analysis** to determine variations in dimensions.
  - **Visual validation** of bounding boxes.
- Converted **CSV annotations to YOLO format**:
  - Normalized bounding box coordinates.
  - Mapped class labels to numerical indices.

### 2. Model Selection & Training
- **YOLOv8** was chosen for its high-speed inference and accuracy in object detection.
- Fine-tuned a **pretrained YOLOv8 model** on the dataset.
- **Training details:**
  - **Epochs:** 50
  - **Batch size:** 8
  - **Image size:** 416x416
  - **Loss function:** YOLO standard loss
- The trained model weights were saved for inference.

### 3. Model Inference & Evaluation
- The trained model was used to predict bounding boxes on test images.
- Final predictions were compiled into a submission **CSV file** containing:
  - **Filename**
  - **Defect class (hotspot, diode, or no defect)**
  - **Bounding box coordinates (xmin, ymin, xmax, ymax)**

## Repository Structure
```
├── data/                  # Dataset files (not included in repo)
│   ├── Train/             # Training images and annotations
│   ├── Test/              # Test images
├── yolo_dataset/          # YOLO formatted dataset
├── models/                # Trained model weights
├── notebooks/             # Jupyter Notebook for the solution
│   ├── Zelestra.ipynb     # Notebook with full solution
├── scripts/               # Python scripts for processing and training
│   ├── Script.py          # Main training and inference script
├── docs/                  # Documentation files
│   ├── Approach Document.pdf  # Detailed methodology
│   ├── Problem Statement.docx # Problem details
├── submission/            # Output files
│   ├── manish_gupta.csv   # Final submission file
├── requirements.txt       # Required packages
├── README.md              # This file
```

## Installation & Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/PV-Defect-Detection.git
   cd your-repo-name
   ```
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Prepare YOLO dataset**:
   ```bash
   python Script.py
   ```
4. **Train the YOLOv8 model**:
   ```bash
   python Script.py
   ```
5. **Generate predictions**:
   ```bash
   python Script.py
   ```

## Results
- Successfully detected **hotspot defects** and **diode failures** with YOLOv8.
- Generated a structured **CSV submission file** with bounding box predictions.
- Ensured **high-speed inference** while maintaining accuracy.

## Future Improvements
- Experimenting with **data augmentation** techniques.
- Optimizing hyperparameters for better accuracy.
- Exploring **ensemble models** for performance improvement.

## Contributors
- **Manish Gupta** - Developer & Data Scientist

## License
This project is open-source under the **MIT License**.
