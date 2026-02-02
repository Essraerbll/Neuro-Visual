"""
🧠 NEURO-VISUAL ASSISTANT - ANA DEMO 🧠

Bu script, Computer Vision (nesne algılama) ve BCI (beyin sinyalleri) 
simülasyonunu birleştirerek, akıllı oda kontrolünü gösterir.

KONTROLLER:
  - '1', '2', '3'... : Klavyeden nesne seç (BCI sinyalini simüle et)
  - 'c'              : Seçimi temizle
  - 'q'              : Çıkış
"""

import cv2
import sys
from pathlib import Path

# Modülleri import et
sys.path.insert(0, str(Path(__file__).parent))
from camera_vision import CameraVision
from bci_simulator import BCISimulator


class NeuroVisualAssistant:
    """Neuro-Visual Assistant ana sınıfı"""
    
    def __init__(self):
        print("\n" + "="*60)
        print("🧠 NEURO-VISUAL ASSISTANT - DEMO 🧠".center(60))
        print("="*60)
        
        self.vision = CameraVision(model_size='n')
        self.bci = BCISimulator()
        self.running = False
        
    def start(self):
        """Sistemi başlat"""
        self.vision.start_camera(camera_id=0)
        self.running = True
        self.main_loop()
        
    def handle_keyboard_input(self, key):
        """Klavye girdisini işle"""
        if key == ord('q'):
            print("\n🛑 Sistem kapatılıyor...")
            self.running = False
            return
        
        # Sayı tuşları: 1-9 arası nesneleri seç
        if ord('0') <= key <= ord('9'):
            obj_idx = key - ord('0')
            if obj_idx > 0 and obj_idx <= len(self.vision.detected_objects):
                selected_idx = obj_idx - 1
                selected_name = self.vision.detected_objects[selected_idx]['name']
                print(f"\n📌 Seçildi: {selected_name} (İndeks: {selected_idx})")
                self.bci.user_selects_object(selected_idx, confidence=0.90)
            else:
                print(f"❌ Geçersiz nesne numarası")
        
        elif key == ord('c'):
            print("\n✨ Seçim temizlendi")
            self.bci.clear_selection()
    
    def main_loop(self):
        """Ana döngü"""
        print("\n" + "="*60)
        print("KONTROLLER:")
        print("  '1'-'9' : Klavyeden nesne seç")
        print("  'c'     : Seçimi temizle")
        print("  'q'     : Çıkış")
        print("="*60 + "\n")
        
        frame_count = 0
        
        while self.running:
            # Kameradan frame yakala
            if not self.vision.capture_frame():
                print("❌ Kameradan frame alınamadı!")
                break
            
            # Nesneleri algıla
            objects = self.vision.detect_objects(confidence=0.5)
            frame_count += 1
            
            # Seçili nesne varsa bul
            selected_idx = self.bci.get_selected_object()
            
            # Frame'i çiz
            display_frame = self.vision.draw_detections(
                selected_idx=selected_idx,
                highlight_color=(0, 200, 0)  # Yeşil
            )
            
            # Bilgi paneli ekle
            self._draw_info_panel(display_frame, objects, selected_idx)
            
            # Göster
            cv2.imshow("🧠 Neuro-Visual Assistant", display_frame)
            
            # Klavyeden giriş al (1ms bekleme)
            key = cv2.waitKey(1) & 0xFF
            if key != 255:  # 255 = tuş basılmadı
                self.handle_keyboard_input(key)
            
            # Her 30 frame'de bir durumu yazdır
            if frame_count % 30 == 0:
                obj_list = self.vision.get_object_list()
                if obj_list:
                    print(f"📹 Frame: {frame_count} | Algılanan: {', '.join(set(obj_list[:3]))}")
        
        self.cleanup()
    
    def _draw_info_panel(self, frame, objects, selected_idx):
        """Ekrana bilgi paneli çiz"""
        height, width = frame.shape[:2]
        panel_height = 120
        
        # Panel arka planı
        cv2.rectangle(frame, (0, height - panel_height), (width, height), 
                     (30, 30, 30), -1)
        
        # Yazılar
        y_offset = height - panel_height + 30
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Başlık
        cv2.putText(frame, "NEURO-VISUAL ASSISTANT", (10, y_offset),
                   font, 0.7, (0, 255, 255), 2)
        
        # Algılanan nesne sayısı
        obj_count = len(objects)
        cv2.putText(frame, f"Nesneler: {obj_count}", (10, y_offset + 35),
                   font, 0.6, (0, 200, 0), 1)
        
        # Seçili nesne bilgisi
        if selected_idx is not None and selected_idx < len(objects):
            selected_obj = objects[selected_idx]
            status = f"Seçili: {selected_obj['name'].upper()}"
            color = (0, 255, 255)  # Sarı
        else:
            status = "Bekleniyor..."
            color = (100, 100, 100)  # Gri
        
        cv2.putText(frame, status, (width // 2, y_offset + 35),
                   font, 0.6, color, 1)
        
        # Kontroller ipucu
        cv2.putText(frame, "Kontrol: 1-9 (Seç) | C (Temizle) | Q (Çıkış)", 
                   (10, y_offset + 65), font, 0.5, (200, 200, 200), 1)
    
    def cleanup(self):
        """Kaynakları temizle"""
        self.vision.release()
        cv2.destroyAllWindows()
        print("\n✅ Temizlik tamamlandı")


def main():
    """Ana fonksiyon"""
    try:
        assistant = NeuroVisualAssistant()
        assistant.start()
    except Exception as e:
        print(f"\n❌ Hata: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
