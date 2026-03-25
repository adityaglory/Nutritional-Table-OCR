import streamlit as st
import cv2
import numpy as np
import pandas as pd
from main_ui import init_models, process_image_engine

st.set_page_config(page_title="Nutrition AI Scanner", page_icon="🥛", layout="wide")

@st.cache_resource
def load_ai_models():
    return init_models()

yolo, ocr = load_ai_models()

st.title("SMART NUTRITION OCR")
st.markdown("Unggah foto kemasan, AI akan mendeteksi tabel gizi, mengekstrak data, dan melakukan *Cross-Imputation* secara otomatis.")

uploaded_files = st.file_uploader("Upload Gambar Kemasan", type=['jpg', 'jpeg', 'png'], accept_multiple_files=True)

if uploaded_files:
    if st.button("Ekstrak Data Gizi", type="primary", use_container_width=True):
        
        all_data = []
        progress_bar = st.progress(0)
        
        for idx, file in enumerate(uploaded_files):
            file_bytes = np.asarray(bytearray(file.read()), dtype=np.uint8)
            img = cv2.imdecode(file_bytes, 1)
            
            with st.spinner(f"Memproses gambar: {file.name}..."):
                data_list, crops = process_image_engine(img, file.name, yolo, ocr)
            
            if crops:
                st.subheader(f"🖼️ Hasil Deteksi: {file.name}")
                cols = st.columns(len(crops))
                for i, (label, crop_img) in enumerate(crops):
                    rgb_crop = cv2.cvtColor(crop_img, cv2.COLOR_BGR2RGB)
                    cols[i].image(rgb_crop, caption=label, use_container_width=True)
                
                all_data.extend(data_list)
            else:
                st.warning(f"❌ Tabel gizi tidak ditemukan pada {file.name}.")
                
            progress_bar.progress((idx + 1) / len(uploaded_files))
            
        if all_data:
            st.success("✅ Ekstraksi teks selesai! Menganalisis metadata...")
            df = pd.DataFrame(all_data)
            
            def sync_serving(x):
                mode = x.dropna().mode()
                return x.fillna(mode.iloc[0]) if not mode.empty else x

            def sync_container(x):
                mode = x.dropna().mode()
                return x.fillna(mode.iloc[0]) if not mode.empty else x.fillna(1.0)

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

            st.dataframe(df_final, use_container_width=True)
            
            csv = df_final.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📥 Download Database (CSV)",
                data=csv,
                file_name='nutrition_database_ui.csv',
                mime='text/csv',
            )