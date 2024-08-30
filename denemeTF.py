import os
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest

# Belgeleri okuma fonksiyonu
def belgeleri_oku(dosya_yolu):
    with open(dosya_yolu, 'r', encoding='utf-8') as file:
        belgeler = [line.strip() for line in file]
    return belgeler

ana_klasor_yolu = r"C:\Users\Semiha\Desktop\makine_proje"

# Kategori isimleri ve dosya adları
kategoriler = ["toplam_negatif_preprocessed", "toplam_pozitif_preprocessed"]

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

    # TF-IDF vektörizasyonu
    vectorizer = TfidfVectorizer()
    egitim_tfidf = vectorizer.fit_transform(tum_egitim_belgeleri)
    test_tfidf = vectorizer.transform(tum_test_belgeleri)

    # Chi-Square öznitelik seçimi
    chi2_values, p_values = chi2(egitim_tfidf, egitim_classes)

    # En ayırt edici öznitelikleri seçme
    for k in [250, 500, 1000, 2500, 5000]:
        selector = SelectKBest(chi2, k=k)
        egitim_tfidf_selected = selector.fit_transform(egitim_tfidf, egitim_classes)
        test_tfidf_selected = selector.transform(test_tfidf)

        selected_features = vectorizer.get_feature_names_out()[selector.get_support()]

        # DataFrame oluşturma
        egitim_tfidf_df = pd.DataFrame(egitim_tfidf_selected.toarray(), columns=selected_features)
        test_tfidf_df = pd.DataFrame(test_tfidf_selected.toarray(), columns=selected_features)

        # Sınıfları ekleme
        egitim_tfidf_df['Class'] = egitim_classes
        test_tfidf_df['Class'] = test_classes

        # CSV dosyalarına yazma
        egitim_csv_yolu = os.path.join(fold_klasor_yolu, f"egitim_tfidf_sonuclar_{k}.csv")
        test_csv_yolu = os.path.join(fold_klasor_yolu, f"test_tfidf_sonuclar_{k}.csv")
        egitim_tfidf_df.to_csv(egitim_csv_yolu, index=False, encoding='utf-8-sig')
        test_tfidf_df.to_csv(test_csv_yolu, index=False, encoding='utf-8-sig')

        print(f"Eğitim sonuçları {egitim_csv_yolu} dosyasına yazıldı.")
        print(f"Test sonuçları {test_csv_yolu} dosyasına yazıldı.")
