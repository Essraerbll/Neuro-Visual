"""
...
BCI Simülatör Modülü
Gerçek EEG verileri gelene kadar, kullanıcının seçimini simüle eder
İleriye dönük: Gerçek EEG sinyallerinden P300/SSVEP özelliklerini çıkarmak için hazır
"""

import numpy as np
from typing import Optional


class BCISimulator:
    """Beyin sinyalini simüle eden sınıf"""
    
    def __init__(self):
        self.selected_object_idx = None
        self.selection_active = False
        self.confidence = 0.0
        
    def user_selects_object(self, object_idx: int, confidence=0.95):
        """
        Simüle: Kullanıcı object_idx'li nesneyi seçti
        
        Gerçek hayatta: EEG başlığından P300/SSVEP sinyali alırız ve
        hangi nesneye bakıldığını şu fonksiyona geçeriz
        """
        self.selected_object_idx = object_idx
        self.confidence = confidence
        self.selection_active = True
        print(f"🧠 BCI Sinyali: Nesne #{object_idx} seçildi (Güven: {confidence:.2f})")
        
    def clear_selection(self):
        """Seçimi temizle"""
        self.selected_object_idx = None
        self.confidence = 0.0
        self.selection_active = False
        
    def get_selected_object(self) -> Optional[int]:
        """Seçili nesnenin indeksini döndür"""
        return self.selected_object_idx if self.selection_active else None
    
    def get_confidence(self) -> float:
        """Seçim güven skorunu döndür (0-1)"""
        return self.confidence


class BCIDataGenerator:
    """
    Gerçek EEG verisini simüle eden sınıf
    (Tez çalışmasında bunu gerçek MATLAB/MNE-Python koduna çevirirsin)
    """
    
    def __init__(self, sampling_rate=250, n_channels=8):
        """
        Args:
            sampling_rate: Hz cinsinden örnekleme oranı (típik: 250Hz)
            n_channels: EEG kanal sayısı (típik: 8, 16, 32)
        """
        self.sampling_rate = sampling_rate
        self.n_channels = n_channels
        self.eeg_data = np.zeros((n_channels, sampling_rate))
        
    def generate_noise(self, duration=1.0) -> np.ndarray:
        """Arka plan EEG gürültüsü üret"""
        n_samples = int(duration * self.sampling_rate)
        # Gauss gürültü + 1/f ruzgârı (pink noise)
        noise = np.random.randn(self.n_channels, n_samples)
        return noise
    
    def generate_p300_event(self, event_time=0.3, amplitude=10):
        """
        P300 dalgasını simüle et
        Odaklanma sırasında ~300ms sonra maksimum tepki görülür
        """
        n_samples = int(1.0 * self.sampling_rate)
        t = np.linspace(0, 1.0, n_samples)
        
        # Gaussian pulse
        p300_wave = np.zeros((self.n_channels, n_samples))
        for ch in range(self.n_channels):
            p300_wave[ch] = amplitude * np.exp(-((t - event_time)**2) / 0.02)
        
        return p300_wave
    
    def generate_ssvep_response(self, target_frequency=12, duration=1.0, amplitude=5):
        """
        SSVEP (Steady-State Visual Evoked Potential) sinyali üret
        Kullanıcı belirli frekansda yanıp sönen nesneye baktığında,
        beyni o frekansta sinyal üretir
        """
        n_samples = int(duration * self.sampling_rate)
        t = np.linspace(0, duration, n_samples)
        
        ssvep = np.zeros((self.n_channels, n_samples))
        for ch in range(self.n_channels):
            ssvep[ch] = amplitude * np.sin(2 * np.pi * target_frequency * t)
        
        return ssvep
    
    def get_sample(self) -> np.ndarray:
        """Bir örnek EEG veri parçası döndür"""
        return self.eeg_data


if __name__ == "__main__":
    # Test
    bci_sim = BCISimulator()
    
    print("🧠 BCI Simülatörü Test")
    print("-" * 40)
    
    # Nesne seçimi simüle et
    bci_sim.user_selects_object(object_idx=2, confidence=0.92)
    print(f"Seçili nesne: {bci_sim.get_selected_object()}")
    print(f"Güven: {bci_sim.get_confidence():.2f}")
    
    bci_sim.clear_selection()
    print(f"Seçimi temizledik: {bci_sim.get_selected_object()}")
    
    print("\n📊 EEG Veri Üretimi Test")
    print("-" * 40)
    gen = BCIDataGenerator()
    p300 = gen.generate_p300_event()
    print(f"P300 şekli: {p300.shape}")
    
    ssvep = gen.generate_ssvep_response(target_frequency=15)
    print(f"SSVEP şekli: {ssvep.shape}")
