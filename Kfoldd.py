import os
from sklearn.model_selection import KFold

# Belgeleri okuma fonksiyonu
def belgeleri_oku(dosya_yolu):
    # Dosya yolu yoksa, hata mesajı ver ve boş liste döndür
    if not os.path.exists(dosya_yolu):
        print(f"Dosya bulunamadı: {dosya_yolu}")
        return []
    # Dosya mevcutsa, dosyayı oku ve her satırı belgeler listesine ekle
    with open(dosya_yolu, 'r', encoding='utf-8') as file:
        belgeler = [line.strip() for line in file]
    return belgeler

# Belgeleri k-fold cross validation ile böl
def kfold_ayir_ve_yaz(ana_klasor_yolu, kategori, n_splits=5):
    # Kategoriye ait dosya yolunu oluştur
    dosya_yolu = os.path.join(ana_klasor_yolu, f"{kategori}.txt")
    # Belgeleri oku
    belgeler = belgeleri_oku(dosya_yolu)
    # Belgeler boşsa, fonksiyondan çık
    if not belgeler:
        return
    
    # K-fold cross validation objesini oluştur
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fold_num = 1
    
    # K-fold split'i uygula
    for train_index, test_index in kf.split(belgeler):
        # Eğitim ve test belgelerini ayır
        train_belgeler = [belgeler[i] for i in train_index]
        test_belgeler = [belgeler[i] for i in test_index]
        
        # Fold klasörü adı oluştur
        fold_klasor_adi = f"Fold{fold_num}"
        # Fold klasörü yolu oluştur
        fold_klasor_yolu = os.path.join(ana_klasor_yolu, fold_klasor_adi)
        
        # Fold klasörü mevcut değilse, oluştur
        if not os.path.exists(fold_klasor_yolu):
            os.makedirs(fold_klasor_yolu)
        
        # Eğitim dosyasının adını oluştur
        egitim_dosya_adi = os.path.join(fold_klasor_yolu, f"eğitim_{kategori}.txt")
        # Test dosyasının adını oluştur
        test_dosya_adi = os.path.join(fold_klasor_yolu, f"test_{kategori}.txt")
        
        # Eğitim belgelerini dosyaya yaz
        with open(egitim_dosya_adi, 'w', encoding='utf-8') as file:
            file.write("\n".join(train_belgeler))
        
        # Test belgelerini dosyaya yaz
        with open(test_dosya_adi, 'w', encoding='utf-8') as file:
            file.write("\n".join(test_belgeler))
        
        # Fold numarasını artır
        fold_num += 1
        print(f"{fold_klasor_yolu} için {kategori} kategorisi eğitim ve test dosyaları oluşturuldu.")

# Ana klasör yolu ve kategori adları
ana_klasor_yolu = r"C:\Users\Semiha\Desktop\veri seti"
kategori_adlari = [
    "DENİZ", "HAVUZ", "Hizmet", "KONUM VE CEVRE", "ODA", "PERSONEL", "YEMEK"
]

# Her kategori için k-fold cross validation işlemi uygulamak
for kategori in kategori_adlari:
    kfold_ayir_ve_yaz(ana_klasor_yolu, kategori)
