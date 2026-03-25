import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd

def init_models():
    yolo_model = YOLO('runs/detect/model_gizi/weights/best.pt').to('cuda:0') 
    ocr_reader = easyocr.Reader(['id', 'en'], gpu=True)
    return yolo_model, ocr_reader

def sharpen_document(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(gray)

def find_number(text, keyword, window=30, max_val=None, val_type='macro'):
    match = re.search(keyword, text)
    if not match: return None
    start_idx = match.end()
    context_text = text[start_idx : start_idx + window]
    nums = re.findall(r'(\d+(?:\.\d+)?)\s*([a-z%]+)?', context_text)
    for num_str, unit in nums:
        if unit and '%' in unit: continue
        if val_type == 'macro' and len(num_str) >= 2 and (num_str.endswith('8') or num_str.endswith('9')) and not unit:
            num_str = num_str[:-1]
        val = float(num_str)
        if val_type == 'macro' and val > 100: continue 
        if max_val and val > max_val: continue 
        return val
    return None

def find_servings_per_container(text):
    match = re.search(r'(\d+(?:\.\d+)?)\s*(sajian per kemasan|servings per container)', text)
    if match: 
        val = float(match.group(1))
        if val <= 50: return val 
    match = re.search(r'(\d+)\s*.{0,5}?(sajian|serving)', text)
    if match:
        val = float(match.group(1))
        if 0 < val <= 50: return val
    match = re.search(r'(jumlah sajian|sajian|serving).{0,5}?(\d+)', text)
    if match:
        val = float(match.group(2))
        if 0 < val <= 50: return val
    return None

def extract_nutrition(img_array, label_name, ocr_reader):
    processed_img = sharpen_document(img_array)
    ocr_results = ocr_reader.readtext(processed_img, detail=0)
    full_text = " ".join(ocr_results).lower()
    
    servings_per_container = find_servings_per_container(full_text)
    serving_size = find_number(full_text, r'(serving size|takaran saji|takaran)', window=40, max_val=1000, val_type='serving')
    calories = find_number(full_text, r'(energi total|total energy|kalori)', window=40, max_val=2000, val_type='calorie')
    carbs = find_number(full_text, r'(karbohidrat|carbohydrate)', window=30, max_val=100, val_type='macro')
    sugar = find_number(full_text, r'(gula\w*|sugars?)', window=30, max_val=100, val_type='macro') 
    protein = find_number(full_text, r'(protein)', window=30, max_val=100, val_type='macro')
    fat = find_number(full_text, r'(lemak total|total fat)', window=30, max_val=100, val_type='macro')

    if serving_size or calories or carbs or protein or fat:
        return {
            "Object_Name": label_name,
            "Serving_Size_g": serving_size,
            "Servings_per_Container": servings_per_container,
            "Calories_per_Serving": calories,
            "Carbs_per_Serving_g": carbs,
            "Sugar_per_Serving_g": sugar, 
            "Protein_per_Serving_g": protein,
            "Fat_per_Serving_g": fat
        }
    return None

def process_image_engine(img, filename, yolo_model, ocr_reader):
    results = yolo_model(img, conf=0.4, verbose=False) 
    boxes = results[0].boxes 
    
    extracted_data_list = []
    crop_images = []

    if len(boxes) > 0:
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = yolo_model.names[int(box.cls[0])]
            cropped_img = img[y1:y2, x1:x2]
            
            label = f"{filename} - Table {i+1} ({class_name})"
            data_row = extract_nutrition(cropped_img, label, ocr_reader)
            
            if data_row: 
                # INJEKSI FILE_NAME UNTUK LOGIKA GROUP BY NANTI
                data_row['File_Name'] = filename
                extracted_data_list.append(data_row)
                crop_images.append((label, cropped_img))

    return extracted_data_list, crop_images