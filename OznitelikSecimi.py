import os
import pandas as pd
from sklearn.feature_selection import SelectKBest, chi2

# Fold1 klasör yolunu belirleyin
folder_path = 'C:/Users/Semiha/Desktop/makine_proje_YENI/Fold5'

# Seçilecek öznitelik sayıları
k_values = [250, 500, 1000, 2500, 5000]

# Eğitim CSV dosyasının adını belirleyin
egitim_csv = os.path.join(folder_path, 'egitim_tfidf_sonuclar.csv')

# Eğitim CSV dosyasını yükle
if os.path.exists(egitim_csv):
    egitim_df = pd.read_csv(egitim_csv)
else:
    print(f"Uygun CSV dosyası {folder_path} için bulunamadı.")
    exit()

# Eğitim verilerini ayırın
X_egitim = egitim_df.drop(columns=['Class'])
y_egitim = egitim_df['Class']

# Her bir k değeri için öznitelik seçimi yapın
for k in k_values:
    # Chi-square kullanarak en iyi k özniteliği seçin
    selector = SelectKBest(chi2, k=k)
    X_new = selector.fit_transform(X_egitim, y_egitim)

    # Seçilen özniteliklerin isimlerini alın
    mask = selector.get_support()
    selected_features = X_egitim.columns[mask]

    # Yeni dataframe oluşturun ve seçilen öznitelikleri ekleyin
    selected_df = pd.DataFrame(X_new, columns=selected_features)
    selected_df['Class_Label'] = y_egitim.values

    # Yeni dosya adını belirleyin
    output_txt = os.path.join(folder_path, f'selected_features_{k}.txt')

    # Yeni dataframe'i dosyaya yazdırın (sadece öznitelik isimleri)
    with open(output_txt, 'w') as f:
        for feature in selected_features:
            f.write(f"{feature}\n")
    print(f"{folder_path} için {k} öznitelikli dosya yazıldı: {output_txt}")

print("Fold1 için öznitelik seçimi tamamlandı.")
