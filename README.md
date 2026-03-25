# Smart Nutrition OCR & Data Extraction Pipeline

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-2496ED?style=flat&logo=docker&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat&logo=opencv&logoColor=white)

An end-to-end, GPU-accelerated Computer Vision pipeline designed to extract, validate, and structure nutritional data from noisy, high-glare product packaging. This project features a dual-interface architecture (Streamlit Web UI & CLI Batch Processor) and is fully containerized using Docker with NVIDIA CUDA support.

## Key Features

* GPU-Accelerated Text Extraction: Utilizes a fine-tuned custom YOLO model for precise table localization and EasyOCR (with Eagle Eye configuration) to extract bilingual text from challenging image conditions.
* Smart Data Engineering:
  * Fuzzy Regex Matching: Tolerates OCR typos and spacing errors.
  * Atwater System Validation: Automatically recalculates and validates missing macronutrients (Carbohydrates) based on total calories, protein, and fat.
  * Cross-Box Imputation: Intelligently syncs missing metadata (like serving sizes) across multiple detected tables within the same image.
* Dual Interface Architecture:
  * Interactive Streamlit Web UI for single/multi-image analysis.
  * Robust CLI Engine for high-throughput batch processing of entire dataset folders.
* Production-Ready Deployment: Fully containerized with a PyTorch/CUDA base image, ensuring zero-dependency setup and consistent environments across different hardwares.

---

## Project Architecture

```text
Smart-Nutrition-OCR
 ┣ runs/detect/model_gizi/weights/  # Custom YOLO weights (best.pt)
 ┣ app.py                           # Streamlit Web UI Frontend
 ┣ main_ui.py                       # Core Extraction Engine (API for UI)
 ┣ main_cli.py                      # CLI Script for Batch Processing
 ┣ requirements.txt                 # Python Dependencies
 ┣ Dockerfile                       # Containerization Recipe
 ┗ README.md
```

## Installation & Setup

### Option A: Using Docker (Recommended)
Ensure you have Docker and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed to utilize GPU acceleration.

1. Build the Docker Image:
   ```bash
   docker build -t nutrition-ocr-app .
   ```
2. Run the Container:
   ```bash
   docker run --gpus all -p 8501:8501 nutrition-ocr-app
   ```
3. Open your browser and navigate to http://localhost:8501

### Option B: Local Setup (Virtual Environment)
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/Smart-Nutrition-OCR.git](https://github.com/yourusername/Smart-Nutrition-OCR.git)
   cd Smart-Nutrition-OCR
   ```
2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage Guide

1. Web Interface (Streamlit)
To launch the interactive UI for dragging and dropping images:
```bash
streamlit run app.py
```
2. Command Line Interface (Batch Processing)
To process a single image:
```bash
python main_cli.py sample_image.jpg
```
To process an entire folder of images automatically:
```bash
python main_cli.py path/to/your/dataset_folder/
```
*The CLI will automatically compile all extracted data into a clean nutrition_database.csv.*
   
