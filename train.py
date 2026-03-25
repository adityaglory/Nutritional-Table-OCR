from ultralytics import YOLO

def main():
    print("⏳ Memuat pre-trained model YOLOv8n...")
    # Kita menggunakan model 'nano' (yolov8n.pt) sebagai dasar
    model = YOLO('yolov8n.pt') 

    print("🚀 Memulai proses Fine-Tuning di Local GPU...")
    # Memulai pelatihan (Transfer Learning)
    results = model.train(
        data='dataset/data.yaml', # Lokasi buku panduan dataset kita
        epochs=50,                # Jumlah putaran belajar (50 sudah cukup bagus untuk awal)
        imgsz=640,                # Resolusi gambar yang kita set di Roboflow
        device=0,                 # MEMAKSA menggunakan NVIDIA GPU (CUDA device 0)
        batch=16,                 # Jumlah gambar yang diproses sekali jalan
        name='model_gizi',        # Nama folder tempat hasil latihan disimpan
        workers=4                 # Mempercepat loading data menggunakan CPU
    )

    print("\n🎉 TRAINING SELESAI!")
    print("Model AI buatanmu yang sudah pintar tersimpan di: runs/detect/model_gizi/weights/best.pt")

if __name__ == '__main__':
    main()