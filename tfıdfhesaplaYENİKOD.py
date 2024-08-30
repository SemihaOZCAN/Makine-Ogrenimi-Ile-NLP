import os
import numpy as np
import pandas as pd

# Belgeleri okuma fonksiyonu
def belgeleri_oku(dosya_yolu):
    with open(dosya_yolu, 'r', encoding='utf-8') as file:
        belgeler = [line.strip() for line in file]
    return belgeler

# TF, IDF ve TF-IDF değerlerini hesapla
def tf_hesapla(belge):
    kelimeler = belge.split(" ")
    terim_frekansi = {}
    for kelime in kelimeler:
        terim_frekansi[kelime] = terim_frekansi.get(kelime, 0) + 1
    return terim_frekansi

def idf_hesapla(belgeler, kelime):
    kelimeyi_iceren_belge_sayisi = sum(1 for belge in belgeler if kelime in belge)
    if kelimeyi_iceren_belge_sayisi > 0:
        return np.log10(len(belgeler) / kelimeyi_iceren_belge_sayisi)
    else:
        return 0

def tf_idf_hesapla(belgeler, terim_idf):
    tfidf_matrisi = np.zeros((len(belgeler), len(terim_idf)))
    
    for i, belge in enumerate(belgeler):
        tf = tf_hesapla(belge)
        for j, (kelime, idf) in enumerate(terim_idf.items()):
            tfidf_matrisi[i, j] = tf.get(kelime, 0) * idf
            
    return tfidf_matrisi

ana_klasor_yolu = r"C:\Users\Semiha\Desktop\ALTI KATEGORI VERI SETI"

# Kategori isimleri ve dosya adları
kategoriler = [ "DENİZ-Deniz (Dalga durumu, zemin durumu, Temizliği)","DENİZ-PLAJ-Plaj Temizliği","DENİZ-PLAJ-Plajın İmkanları (Snack bar, şezlong)","HAVUZ-Havuzların Boyutu","HAVUZ-Havuzların sayısı,çeşitliliği","HAVUZ-Havuzların temizliği","HİZMET-Eğlence (Animasyon, Etkinlik Çeşitliliği,Canlı Müzik)","HİZMET-spa-sauna-Fitness","KONUM VE CEVRE-Otel Bahçesi-Doğa güzellikleri","KONUM VE CEVRE-Otel Konumu-Ulaşım kolaylığı","ODA-oda boyutu","ODA-oda temizliği (banyo, tuvalet, yatak)","ODA-odanın Olanakları (Duş sorunu, Sıcak soğuk su sorunu, klima sorunu, Sineklik,koltuk)","ODA-Yatak (Yatak Kapasitesi, Yatak Kalitesi)","YEMEK-Fiyat-performans dengesi","YEMEK-Masa Düzeni ve Temizliği","YEMEK-Snack Bar- Bar","YEMEK-Yemek Çeşitliliği ,Açık Büfe Çeşitliliği","YEMEK-Yemek Kalitesi, Yemek Lezzeti"]

# Ana klasördeki tüm fold klasörlerini listeleme
fold_klasorleri = [klasor for klasor in os.listdir(ana_klasor_yolu) if os.path.isdir(os.path.join(ana_klasor_yolu, klasor))]

for fold_klasor in fold_klasorleri:
    fold_klasor_yolu = os.path.join(ana_klasor_yolu, fold_klasor)

    # Tüm belgeleri ve sınıfları depolamak için listeler
    tum_egitim_belgeleri = []
    tum_test_belgeleri = []
    egitim_classes = []
    test_classes = []

    for class_value, kategori in enumerate(kategoriler):
        # Eğitim ve test dosyalarının adlarını tanımlama
        egitim_dosya_adi = f"eğitim_{kategori}.txt"
        test_dosya_adi = f"test_{kategori}.txt"

        # Eğitim ve test dosyalarını okuma
        egitim_belgeleri = belgeleri_oku(os.path.join(fold_klasor_yolu, egitim_dosya_adi))
        test_belgeleri = belgeleri_oku(os.path.join(fold_klasor_yolu, test_dosya_adi))

        # Belgeleri ve sınıfları birleştirme
        tum_egitim_belgeleri.extend(egitim_belgeleri)
        tum_test_belgeleri.extend(test_belgeleri)
        egitim_classes.extend([class_value] * len(egitim_belgeleri))
        test_classes.extend([class_value] * len(test_belgeleri))

    # Eğitim belgeleri üzerinden IDF değerlerini hesaplama
    terim_idf = {}

    for belge in tum_egitim_belgeleri:
        kelimeler = set(belge.split())
        for kelime in kelimeler:
            terim_idf[kelime] = idf_hesapla(tum_egitim_belgeleri, kelime)

    # TF-IDF matrislerini hesaplama
    egitim_tfidf = tf_idf_hesapla(tum_egitim_belgeleri, terim_idf)
    test_tfidf = tf_idf_hesapla(tum_test_belgeleri, terim_idf)

    # DataFrame oluşturma
    egitim_tfidf_df = pd.DataFrame(egitim_tfidf)
    test_tfidf_df = pd.DataFrame(test_tfidf)

    # Sınıfları ekleme
    egitim_tfidf_df['Class_Label'] = egitim_classes
    test_tfidf_df['Class_Label'] = test_classes

    # DataFrame'e öznitelik isimlerini ekleme
    oznitelik_isimleri = list(terim_idf.keys())
    egitim_tfidf_df.columns = oznitelik_isimleri + ['Class_Label']
    test_tfidf_df.columns = oznitelik_isimleri + ['Class_Label']

    # CSV dosyalarına yazma
    egitim_csv_yolu = os.path.join(fold_klasor_yolu, "egitim_tfidf_sonuclar.csv")
    test_csv_yolu = os.path.join(fold_klasor_yolu, "test_tfidf_sonuclar.csv")
    egitim_tfidf_df.to_csv(egitim_csv_yolu, index=False, encoding='utf-8-sig')
    test_tfidf_df.to_csv(test_csv_yolu, index=False, encoding='utf-8-sig')

    print(f"Eğitim sonuçları {egitim_csv_yolu} dosyasına yazıldı.")
    print(f"Test sonuçları {test_csv_yolu} dosyasına yazıldı.")
