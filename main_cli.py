import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd
import os
import argparse

print("Loading AI system (YOLO + EasyOCR GPU + Geometry)...")
yolo_model = YOLO('runs/detect/model_gizi/weights/best.pt').to('cuda:0') 
ocr_reader = easyocr.Reader(['id', 'en'], gpu=True)

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

def extract_nutrition(img_array, label_name, save_filename):
    cv2.imwrite(save_filename, img_array)
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
        print(f" [{label_name}] Saji: {serving_size} | Kemasan: {servings_per_container} | Kal: {calories} | Carbs: {carbs}g | Sugar: {sugar}g | Prot: {protein}g | Fat: {fat}g")
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

def process_pipeline(image_path, filename):
    print(f"\n--- PROCESSING IMAGE: {filename} ---")
    img = cv2.imread(image_path)
    if img is None: 
        print(f"Error: Cannot read image '{image_path}'. Skipping...")
        return

    results = yolo_model(img, conf=0.4, verbose=False) 
    boxes = results[0].boxes 
    
    extracted_data_list = []

    if len(boxes) > 0:
        print(f" YOLO detected {len(boxes)} nutrition table(s). Extracting text...")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = yolo_model.names[int(box.cls[0])]
            cropped_img = img[y1:y2, x1:x2]
            
            label = f"{filename} - Table {i+1} ({class_name})"
            
            safe_filename = filename.replace(".", "_")
            data_row = extract_nutrition(cropped_img, label, f"crop_{safe_filename}_{i+1}_{class_name}.jpg")
            
            if data_row: 
                data_row['File_Name'] = filename
                extracted_data_list.append(data_row)

    if extracted_data_list:
        df = pd.DataFrame(extracted_data_list) 
        
        print("\n --- RUNNING CALCULATIONS ---")
        
        def sync_serving(x):
            mode = x.dropna().mode()
            if not mode.empty:
                return x.fillna(mode.iloc[0])
            return x

        def sync_container(x):
            mode = x.dropna().mode()
            if not mode.empty:
                return x.fillna(mode.iloc[0])
            return x.fillna(1.0)

        df['Serving_Size_g'] = df.groupby('File_Name')['Serving_Size_g'].transform(sync_serving)
        df['Servings_per_Container'] = df.groupby('File_Name')['Servings_per_Container'].transform(sync_container)
                
        for index, row in df.iterrows():
            multiplier = df.at[index, 'Servings_per_Container']
            
            if pd.notna(row.get('Calories_per_Serving')):
                df.at[index, 'Total_Calories_Container'] = round(row['Calories_per_Serving'] * multiplier, 1)
            if pd.notna(row.get('Carbs_per_Serving_g')): 
                df.at[index, 'Total_Carbs_Container_g'] = round(row['Carbs_per_Serving_g'] * multiplier, 1)
            if pd.notna(row.get('Sugar_per_Serving_g')):
                df.at[index, 'Total_Sugar_Container_g'] = round(row['Sugar_per_Serving_g'] * multiplier, 1)
            if pd.notna(row.get('Protein_per_Serving_g')):
                df.at[index, 'Total_Protein_Container_g'] = round(row['Protein_per_Serving_g'] * multiplier, 1)
            if pd.notna(row.get('Fat_per_Serving_g')):
                df.at[index, 'Total_Fat_Container_g'] = round(row['Fat_per_Serving_g'] * multiplier, 1)

        df = df.astype(object) 
        df.fillna("N/A", inplace=True) 

        clean_columns = [
            "Object_Name", "Serving_Size_g", "Servings_per_Container",
            "Total_Calories_Container", "Total_Carbs_Container_g", "Total_Sugar_Container_g", 
            "Total_Protein_Container_g", "Total_Fat_Container_g"
        ]
        
        for col in clean_columns:
            if col not in df.columns: df[col] = "N/A"
                
        df_final = df[clean_columns]

        print("\n--- FINAL TABULAR DATA (PANDAS DATAFRAME) ---")
        print(df_final.to_markdown(index=False))

        csv_filename = "nutrition_database.csv"
        df_final.to_csv(csv_filename, mode='a' if os.path.exists(csv_filename) else 'w', header=not os.path.exists(csv_filename), index=False)
        print(f"\n--- Data successfully saved to '{csv_filename}' ---") 
    else:
        print("\n No valid nutrition data found.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Nutrition OCR Batch Pipeline")
    parser.add_argument("path", help="Path ke file gambar tunggal ATAU path ke folder berisi gambar")
    args = parser.parse_args()
    
    target_path = args.path

    if os.path.isfile(target_path):
        filename = os.path.basename(target_path)
        process_pipeline(target_path, filename)
        
    elif os.path.isdir(target_path):
        valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
        print(f"\n Membaca folder: {target_path}")
        
        for file in os.listdir(target_path):
            if file.lower().endswith(valid_extensions):
                full_path = os.path.join(target_path, file)
                process_pipeline(full_path, file)
    else:
        print(f"Error: Path '{target_path}' tidak ditemukan. Pastikan file/folder ada.")