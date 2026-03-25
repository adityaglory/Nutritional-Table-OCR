import cv2
import easyocr
import re
from ultralytics import YOLO
import pandas as pd
import os

# ==========================================
# 1. INISIALISASI MODEL GLOBAL
# ==========================================
print("⏳ Memuat model YOLOv8 dan EasyOCR...")
yolo_model = YOLO('yolov8n.pt') 
ocr_reader = easyocr.Reader(['id', 'en'], gpu=False)
print("✅ Model siap digunakan!\n")

# ==========================================
# 2. FUNGSI EKSTRAKSI (Fitur Pengereman "Stop-Word")
# ==========================================
def extract_nutrition(img_array, label_name, save_filename):
    cv2.imwrite(save_filename, img_array)
    
    ocr_results = ocr_reader.readtext(img_array, detail=0)
    full_text = " ".join(ocr_results).lower()
    
    serving_size = None
    protein_val = None

    # --- LOGIKA 1: MENCARI TAKARAN SAJI ---
    saji_match = re.search(r'(serving size|seving stze|takaran saji|takaran)', full_text)
    if saji_match:
        start_idx = saji_match.end()
        context_text = full_text[start_idx : start_idx + 40]
        nums = re.findall(r'(\d+(?:\.\d+)?)\s*(ml|g|gram)?', context_text)
        if nums:
            serving_size = float(nums[0][0])

    # --- LOGIKA 2: MENCARI PROTEIN (Dengan Pengereman) ---
    prot_match = re.search(r'protein', full_text)
    if prot_match:
        start_idx = prot_match.end()
        # Ambil 50 karakter ke depan
        context_text = full_text[start_idx : start_idx + 50]
        
        # 🛑 PENGEREMAN OTOMATIS: 
        # Jika ada kata "karbohidrat" atau "karbo", potong teksnya sampai di situ saja!
        # Ini mencegah AI "kebablasan" membaca baris di bawahnya.
        if 'karbo' in context_text:
            context_text = context_text.split('karbo')[0]
            
        nums = re.findall(r'(\d+(?:\.\d+)?)\s*([a-z%]+)?', context_text)
        
        print(f"   [DEBUG PROTEIN {label_name}] Area teks aman: '{context_text.strip()}'")
        print(f"   [DEBUG PROTEIN {label_name}] Angka yg tertangkap: {nums}")
        
        for num_str, unit in nums:
            if unit and '%' in unit:
                continue
                
            if (num_str.endswith('8') or num_str.endswith('9')) and len(num_str) > 1 and not unit:
                num_str = num_str[:-1]
                
            val = float(num_str)
            if val > 50:
                continue
                
            protein_val = val
            break 

    # --- SANITY CHECK & PENGEMBALIAN DATA ---
    if serving_size and protein_val:
        per_gram = round(protein_val / serving_size, 3)
        print(f"   ✅ Sukses mengekstrak: Takaran {serving_size}, Protein {protein_val}g")
        
        return {
            "Nama_Objek": label_name,
            "File_Sumber": save_filename,
            "Takaran_Saji": serving_size,
            "Protein_g": protein_val,
            "Rasio_Protein_Per_Unit": per_gram
        }
            
    return None

# ==========================================
# 3. PIPELINE UTAMA & PENYIMPANAN TABULAR
# ==========================================
def process_pipeline(image_path):
    print(f"🔄 --- MEMPROSES GAMBAR: {image_path} ---")
    img = cv2.imread(image_path)
    if img is None:
        print(f"❌ Gambar '{image_path}' tidak ditemukan!")
        return

    results = yolo_model(img, conf=0.05) 
    boxes = results[0].boxes 
    
    extracted_data_list = [] # List untuk menampung baris-baris tabel

    if len(boxes) > 0:
        print(f"✅ YOLO menemukan {len(boxes)} objek. Menganalisis potongan...")
        for i, box in enumerate(boxes):
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = yolo_model.names[int(box.cls[0])]
            cropped_img = img[y1:y2, x1:x2]
            
            label = f"Objek {i+1} ({class_name})"
            filename = f"debug_crop_{i+1}_{class_name}.jpg"
            
            # Panggil fungsi ekstraksi yang mengembalikan Dictionary
            data_row = extract_nutrition(cropped_img, label, filename)
            
            if data_row:
                extracted_data_list.append(data_row)

    # Fallback jika YOLO gagal total mengekstrak apa pun
    if not extracted_data_list:
        print("🚀 Menggunakan Fallback (Global Scan)...")
        data_row = extract_nutrition(img, "Gambar Asli Utuh", "debug_fallback.jpg")
        if data_row:
            extracted_data_list.append(data_row)

    # ==========================================
    # 4. EXPORT KE TABULAR (CSV) UNTUK SQL
    # ==========================================
    if extracted_data_list:
        print("\n📊 MENGUBAH DATA MENJADI TABULAR (PANDAS DATAFRAME)...")
        # Membuat tabel dari list data
        df = pd.DataFrame(extracted_data_list) 
        
        # Menampilkan tabel di terminal agar terlihat profesional
        print(df.to_markdown(index=False)) 
        
        # Menyimpan ke file CSV (Siap di-import ke PostgreSQL/MySQL)
        csv_filename = "database_gizi.csv"
        
        # Jika file sudah ada, tambahkan data ke baris paling bawah (append mode)
        if os.path.exists(csv_filename):
            df.to_csv(csv_filename, mode='a', header=False, index=False)
        else:
            df.to_csv(csv_filename, index=False)
            
        print(f"\n💾 Data berhasil disimpan ke '{csv_filename}'")
    else:
        print("\n❌ Gagal menemukan data gizi untuk dimasukkan ke database.")

if __name__ == "__main__":
    # Tes kembali dengan foto aslimu yang ada 2 kotak susu
    nama_file_foto = 'sample.jpg' 
    process_pipeline(nama_file_foto)