import os
import pandas as pd
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

# Klasör yolunu belirleyin
base_folder_path = r'C:\Users\Semiha\Desktop\HER KATEGORI ICIN AYRI CSV SINIFLANDIRMASI\YEMEK ALT KATEGORI'

# Sınıflandırıcıları ve performans metriklerini saklamak için bir sözlük oluşturun
classifiers = {
    'SVM': SVC(),
    'Random Forest': RandomForestClassifier(),
    'Logistic Regression': LogisticRegression()
}

# Her bir sınıflandırıcı için performans metriklerini saklamak için bir sözlük oluşturun
performance = {classifier: {'accuracies': [], 'f1_scores': []} for classifier in classifiers}

# Fold klasörlerini belirleyin
folds = ['Fold1', 'Fold2', 'Fold3', 'Fold4', 'Fold5']

# Her fold için döngüye girin
for fold in folds:
    folder_path = os.path.join(base_folder_path, fold)
    
    # Eğitim ve test CSV dosyalarının adlarını belirleyin
    egitim_csv = os.path.join(folder_path, 'egitim_tfidf_sonuclar.csv')
    test_csv = os.path.join(folder_path, 'test_tfidf_sonuclar.csv')
    
    # CSV dosyalarını yükle
    if os.path.exists(egitim_csv) and os.path.exists(test_csv):
        egitim_df = pd.read_csv(egitim_csv)
        test_df = pd.read_csv(test_csv)
    else:
        print(f"Uygun CSV dosyaları {fold} için bulunamadı.")
        continue

    # Eğitim ve test verilerini ayırın
    X_egitim = egitim_df.drop(columns=['Class_Label'])
    y_egitim = egitim_df['Class_Label']
    X_test = test_df.drop(columns=['Class_Label'])
    y_test = test_df['Class_Label']

    # Her bir sınıflandırıcı için döngüye girin
    for classifier_name, classifier in classifiers.items():
        # Sınıflandırıcıyı eğitin
        classifier.fit(X_egitim, y_egitim)

        # Test verisini kullanarak tahminler yapın
        y_pred = classifier.predict(X_test)

        # Accuracy ve f1-score hesaplayın
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')

        # Performans metriklerini saklayın
        performance[classifier_name]['accuracies'].append(accuracy)
        performance[classifier_name]['f1_scores'].append(f1)

# Performansları bir metin dosyasına yazdırın
performans_file = 'YEMEK ALT KATEGORI performans_sonuclari.txt'
with open(performans_file, 'w') as file:
    file.write("YEMEK ALT KATEGORI İçin Performanslar:\n\n")
    for classifier_name, scores in performance.items():
        mean_accuracy = sum(scores['accuracies']) / len(scores['accuracies'])
        mean_f1_score = sum(scores['f1_scores']) / len(scores['f1_scores'])

        file.write(f'{classifier_name}:\n')
        file.write(f'Mean Accuracy: {mean_accuracy}\n')
        file.write(f'Mean F1 Score: {mean_f1_score}\n\n')

print(f"Performanslar '{performans_file}' dosyasına yazıldı.")
