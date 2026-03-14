# 🧠 Neuro-Visual Assistant
## Nöro-Görsel Asistan: Beyin-Bilgisayar Arayüzü ile Akıllı Çevre Kontrol Sistemi

---

## 📋 Proje Özeti

**Neuro-Visual Assistant**, engelli bireylerin çevrelerindeki nesnelerle **sadece bakarak ve beyin sinyallerini kullanarak** etkileşim kurmasını sağlayan bir akıllı kontrol sistemidir.

### Temel Prensipler

1. **Computer Vision (Görü)**: Kamera, odadaki nesneleri gerçek zamanlı olarak algılar
2. **BCI (Beyin-Bilgisayar Arayüzü)**: EEG sinyallerinden kullanıcının hangi nesneye odaklanıldığını anlar
3. **Yapay Zeka**: Beyin dalgasından (P300/SSVEP) seçimi sınıflandırır
4. **Akıllı Kontrol**: Çevreden nesneleri (ışık, TV, kapı) kontrol eder

---

## 🎯 Gerçek Hayat Kullanım Senaryosu

> *Yatakta yatan, ellerini kullanamayan bir hastanın karşısında tablet vardır.*
> 
> 1. Tablet kamerası odayı görmektedir
> 2. Ekranda "Işık", "Klima", "Yardım Çağır" butonları görünmektedir
> 3. Hasta "Işık" butonuna bakıyor
> 4. EEG başlığı bu seçimi algılıyor
> 5. **Işık otomatik olarak açılıyor!**

---

## 🛠️ Teknoloji Yığını (Tech Stack)

| Katman | Teknoloji | Amaç |
|--------|-----------|------|
| **Görüntü İşleme** | Python + OpenCV + YOLO | Kameradan nesneleri gerçek zamanlı algıla |
| **Sinyal Analizi** | MATLAB / MNE-Python | EEG verisini işle ve gürültüleri temizle |
| **Yapay Zeka** | Scikit-learn / PyTorch | Beyin sinyalinden seçimi tahmin et |
| **Kullanıcı Arayüzü** | Flutter | Mobil arayüz ve nesneleri göster |
| **Haberleşme** | WebSockets / REST API | Backend-Frontend iletişimi |

---

## 📁 Proje Yapısı

```
Neuro-Visual/
├── src/
│   ├── main_demo.py          # Ana demo (Kamera + Seçim)
│   ├── camera_vision.py      # Nesne algılama modülü
│   ├── bci_simulator.py      # BCI simülatörü (Gerçek EEG yerine)
│   ├── eeg_processor.py      # EEG işleme + özellik çıkarımı + baseline sınıflandırıcı
│   └── eeg_dataset_helper.py # EEG veri seti keşfi/katalog scripti
├── data/
│   └── sample_eeg/           # (Gelecek) Örnek EEG veri setleri
├── models/
│   └── trained_models/       # (Gelecek) Eğitilmiş ML modelleri
├── docs/
│   └── thesis_notes.md       # Tez notları
├── requirements.txt          # Python bağımlılıkları
└── README.md                 # Bu dosya
```

---

## 🚀 Kurulum ve Çalıştırma

### 1. Gereksinimler

- Python 3.8+
- Kamera (USB veya dahili)
- Windows / Linux / macOS

### 2. Bağımlılıkları Yükle

```bash
pip install -r requirements.txt
```

**İlk çalıştırmada YOLO modeli (~100MB) otomatik indirilecektir.**

### 3. Demo'yu Çalıştır

```bash
cd src
python main_demo.py
```

### 4. EEG İşleme Testi

```bash
cd src
python eeg_processor.py
```

### 5. EEG Veri Kaynaklarını Listele

```bash
cd src
python eeg_dataset_helper.py --list
python eeg_dataset_helper.py --export-json ..\data\eeg_dataset_catalog.json
```

---

## 🎮 Kontroller (Demo)

| Tuş | İşlem |
|-----|-------|
| **1-9** | Nesneleri seç (BCI sinyalini simüle et) |
| **C** | Seçimi temizle |
| **Q** | Programı kapat |

### Örnek Oturum

```
🧠 NEURO-VISUAL ASSISTANT
============================================================
📹 Frame: 30 | Algılanan: person, chair, laptop
1 tuşuna basıldı
📌 Seçildi: person (İndeks: 0)
c tuşuna basıldı
✨ Seçim temizlendi
```

---

## 📊 Proje Mimarisi (Detaylı)

### Veri Akışı

```
┌──────────────┐
│   Kamera     │
└──────┬───────┘
       │ Video stream
       ▼
┌──────────────────┐
│  YOLO Modeli     │  (Nesne algılama)
│ (YOLOv8n)        │
└──────┬───────────┘
       │ Tespit: [person, cup, lamp]
       ▼
┌──────────────────┐
│ Flutter UI       │  (Ekranda nesneleri göster)
└──────┬───────────┘
       │ Kullanıcı seçim
       ▼
┌──────────────────┐
│ EEG Başlığı      │  (Brain signals)
└──────┬───────────┘
       │ Raw EEG data
       ▼
┌──────────────────────┐
│ Sinyal İşleme        │  (Filter, Feature extraction)
│ (MATLAB/MNE-Python)  │
└──────┬───────────────┘
       │ P300/SSVEP features
       ▼
┌──────────────────┐
│ ML Sınıflandırma │  (Seçi tahmin et)
└──────┬───────────┘
       │ Predicted object
       ▼
┌──────────────────┐
│ Aksiyon          │  (Işığı aç, TV'yi kapat, vb.)
└──────────────────┘
```

---

## 📚 Tez İçin Anahtar Terimler

- **P300**: Odaklanmaya bağlı ERP (Event-Related Potential)
- **SSVEP**: Steady-State Visual Evoked Potential (Titreme yanıtı)
- **BCI (Brain-Computer Interface)**: Beyin-Bilgisayar Arayüzü
- **EEG (Electroencephalography)**: Beyin elektrik aktivitesini ölçme
- **Feature Extraction**: EEG sinyalinden özellik çıkarma
- **Classification**: Sınıflandırma (Hangi nesne seçildi?)

---

## 🔬 İleri Aşamalar (Roadmap)

### Faz 1: Temel Demo ✅ (ŞİMDİ)
- [x] Kamera + Nesne algılama
- [x] Simüle edilmiş BCI seçimi
- [x] Basit UI gösterimi

### Faz 2: Gerçek EEG Entegrasyonu (YAKINDA)
- [ ] Gerçek EEG başlığı bağlantısı (OpenBCI, g.tec, vb.)
- [x] Python ile temel sinyal işleme pipeline (bandpass/notch/epoch/feature)
- [ ] MATLAB / MNE-Python ile ileri sinyal işleme
- [ ] Gerçek veride P300 algılama doğrulaması

### Faz 3: ML Sınıflandırması (SONRA)
- [ ] Training data topla
- [ ] Scikit-learn / PyTorch modeli eğit
- [ ] Tahmin doğruluğunu test et

### Faz 4: Flutter Mobil UI (SON)
- [ ] Profesyonel arayüz
- [ ] WebSocket iletişimi
- [ ] Gerçek ortamda test

---

## 📖 Kaynaklar

### EEG Veri Setleri (Ücretsiz)

- **PhysioNet**: https://physionet.org/ (Bin saat+ EEG)
- **BCI Competition**: http://www.bbci.de/competition/ (Eğitilmiş EEG)
- **OpenNeuro**: https://openneuro.org/ (Açık kaynak nöro veri)

### Kütüphaneler

- **MNE-Python**: EEG işleme (Python)
- **MATLAB Signal Processing**: Sinyal analizi
- **scikit-learn**: Makine öğrenmesi
- **PyTorch**: Derin öğrenme

---

## 👨‍🎓 Akademik Yazarlar

- **Danışman**: Doç. Dr. Emine Elif Tülay
  - Uzmanlik: EEG Analizi, Kognitif Nörobiyoloji
  - E-mail: [etülay@üniversite.edu]

---

## 📝 Lisans

Bu proje akademik araştırma amaçlı geliştirilmektedir.

---

## 💡 Sık Sorulan Sorular

### S: EEG başlığı olmadan test edebilir miyim?
**C:** Evet! Demo'daki `bci_simulator.py`, klavyeyi kullanarak seçimi simüle eder. Böylece işleyişi anlayabilirsiniz.

### S: Hangi EEG başlığı önerirsiniz?
**C:** Başlangıç için:
- **OpenBCI Ultracortex** (~$2000, açık kaynak)
- **Muse 2** (~$300, uygun ama sınırlı)
- **Emotiv EPOC X** (~$800, profesyonel)

### S: YOLO'nun yanında başka nesne algılatma modeli kullanabilir miyim?
**C:** Evet! ResNet, Faster R-CNN, Mask R-CNN hepsi desteklenir. `camera_vision.py` dosyasını değiştirip test edebilirsiniz.

### S: Flutter'ı nasıl bağlarım?
**C:** `src/` klasöründe yeni bir `backend_api.py` oluşturabiliriz; Flask/FastAPI ile REST API sunacak ve Flutter'dan çağıracağız.

---

## 🐛 Sorun Giderme

### Hata: "CUDA out of memory"
**Çözüm:** `camera_vision.py`'de model boyutunu küçültin:
```python
vision = CameraVision(model_size='n')  # 's', 'm' yerine 'n' (nano)
```

### Hata: "Kamera açılamıyor"
**Çözüm:** Kamera numarasını değiştirin:
```python
vision.start_camera(camera_id=1)  # 0 yerine 1
```

---

**Sorular veya öneriler için lütfen bize yazın!** 🚀
