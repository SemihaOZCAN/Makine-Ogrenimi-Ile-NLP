import os

# Ana klasörün yolu
base_folder = "C:/Users/0beyz/OneDrive/Desktop/PERSONEL"

# Alt klasörlerin adları
subfolders = [
    "Ortak_PERSONEL (Personel ilgisi ve hizmeti,Garson Hizmeti,Servis hızı,resepsiyon hizmetleri)",

]

# Çıkış dosyasının yolu
output_file_path = os.path.join(base_folder, "PERSONEL_txt.txt")

# Çıkış dosyasını oluşturup yazma modunda aç
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    for subfolder in subfolders:
        # Alt klasörün tam yolunu oluştur
        subfolder_path = os.path.join(base_folder, subfolder)
        
        # Alt klasördeki tüm txt dosyalarını bul
        for txt_file in os.listdir(subfolder_path):
            if txt_file.endswith(".txt"):
                txt_file_path = os.path.join(subfolder_path, txt_file)
                
                # Txt dosyasını oku ve içeriğini tek bir satıra yaz
                with open(txt_file_path, 'r', encoding='utf-8') as file:
                    content = file.read().replace('\n', ' ')
                    output_file.write(content + '\n')

print("Tüm txt dosyaları başarıyla Hizmet_txt.txt dosyasına yazıldı.")
