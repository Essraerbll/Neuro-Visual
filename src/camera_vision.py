"""
Kamera Modülü - Nesne Algılama İçin
Görev: Kameradan video akışı almak ve YOLO ile nesneleri tespit etmek
"""

import cv2
import numpy as np
from ultralytics import YOLO
from typing import List, Tuple, Dict


class CameraVision:
    """Kamera tabanlı nesne algılama sistemi"""
    
    def __init__(self, model_size='n'):
        """
        Args:
            model_size: YOLO model boyutu ('n'=nano, 's'=small, 'm'=medium)
        """
        print("🎬 YOLO modeli yükleniyor...")
        self.model = YOLO(f'yolov8{model_size}.pt')
        self.cap = None
        self.frame = None
        self.detected_objects = []
        
    def start_camera(self, camera_id=0):
        """Kamera başlat"""
        self.cap = cv2.VideoCapture(camera_id)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        print(f"✅ Kamera {camera_id} başlatıldı")
        
    def capture_frame(self) -> bool:
        """Kameradan bir frame yakala"""
        ret, self.frame = self.cap.read()
        return ret
    
    def detect_objects(self, confidence=0.5) -> List[Dict]:
        """
        YOLO ile nesneleri algıla
        
        Returns:
            Detected objects list: [{'name': 'dog', 'confidence': 0.95, 'box': [x1, y1, x2, y2]}, ...]
        """
        if self.frame is None:
            return []
        
        results = self.model(self.frame, conf=confidence, verbose=False)
        self.detected_objects = []
        
        for r in results:
            for box in r.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = r.names[cls_id]
                
                self.detected_objects.append({
                    'name': cls_name,
                    'confidence': conf,
                    'box': (x1, y1, x2, y2),
                    'center': ((x1 + x2) // 2, (y1 + y2) // 2)
                })
        
        return self.detected_objects
    
    def draw_detections(self, selected_idx=None, highlight_color=(0, 255, 0)):
        """
        Algılanan nesneleri çizdir
        
        Args:
            selected_idx: Seçili nesnenin indeksi (vurgulanacak)
            highlight_color: Seçili nesne için renk (BGR)
        
        Returns:
            Çizilmiş frame
        """
        if self.frame is None:
            return None
        
        display_frame = self.frame.copy()
        
        for idx, obj in enumerate(self.detected_objects):
            x1, y1, x2, y2 = obj['box']
            
            # Seçili nesne ise farklı renk
            if idx == selected_idx:
                color = (0, 255, 255)  # Sarı
                thickness = 3
            else:
                color = highlight_color
                thickness = 2
            
            # Kutu çiz
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
            
            # Etiket ve güven skoru
            label = f"{obj['name']} {obj['confidence']:.2f}"
            font = cv2.FONT_HERSHEY_SIMPLEX
            cv2.putText(display_frame, label, (x1, y1 - 10),
                       font, 0.6, color, 2)
            
            # Seçili nesneyi vurgula
            if idx == selected_idx:
                cv2.putText(display_frame, "SELECTED", (x1, y2 + 25),
                           font, 0.8, (0, 255, 255), 3)
                # Merkeze bir daire çiz
                cv2.circle(display_frame, obj['center'], 5, (0, 255, 255), -1)
        
        return display_frame
    
    def get_object_list(self) -> List[str]:
        """Algılanan nesnelerin adlarını döndür"""
        return [obj['name'] for obj in self.detected_objects]
    
    def release(self):
        """Kamerayı kapat"""
        if self.cap:
            self.cap.release()
            print("🛑 Kamera kapatıldı")


if __name__ == "__main__":
    # Test
    vision = CameraVision(model_size='n')
    vision.start_camera()
    
    print("Test başladı. 'q' tuşu ile çıkın...")
    
    while True:
        if vision.capture_frame():
            objects = vision.detect_objects(confidence=0.5)
            display = vision.draw_detections()
            
            cv2.imshow("Nesne Algılama", display)
            print(f"Algılanan nesneler: {vision.get_object_list()}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        else:
            break
    
    vision.release()
    cv2.destroyAllWindows()
