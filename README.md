========================================================================

PROJE SUNUM VE TEKNİK RAPOR DOSYASI

========================================================================

TAKIM ADI: AB_degil_01
TAKIM ÜYELERİ:
1. Zeynep Verda ASLAN
2. Ayşegül TEKEŞ
3. Semanur ALTINTAŞ
4. Kader YAPRAK

------------------------------------------------------------------------
1. PROJE ÖZETİ 
------------------------------------------------------------------------
İstanbul 2020 emlak verisini kullanarak bir evin özelliklerinden “Adil Değer (Fair Value)” tahmini yapan
ve ilan fiyatını bu değere göre karşılaştırıp kullanıcıya “FIRSAT / NORMAL / PAHALI” yatırım tavsiyesi
veren bir karar destek sistemi geliştirdik. Sistem Streamlit arayüzü üzerinden tek tuşla çalışır.

------------------------------------------------------------------------
2. KURULUM VE ÇALIŞTIRMA 
------------------------------------------------------------------------
1) Proje klasörüne girin:
   cd AB_degil_01
2) Gerekli paketleri kurun:
   pip install -r requirements.txt
3) Arayüzü çalıştırın:
   streamlit run app.py

Not: Eğitilmiş model dosyası models/ klasörü altında hazırdır (model_bundle.pkl).

------------------------------------------------------------------------
3. VERİ ÖN İŞLEME YAKLAŞIMI 
------------------------------------------------------------------------
- Temizlenen Alanlar:
  * Price sütunundaki TL sembolü/nokta/virgül temizlenerek sayısala çevrildi.
  * Available for Loan = Yes (krediye uygun) filtrelemesi uygulandı.
  * Aykırı fiyatlar için makul aralık filtrelemesi yapıldı (çok uç değerler elendi).
  * Eksik değerler: sayısal kolonlarda median ile, kategorik kolonlarda “Unknown” ile dolduruldu.
  * Leakage/gürültü riski yüksek alanlar (id/url/image/title/description vb.) modele alınmadı.

- Eklenen Yeni Özellikler (Feature Engineering):
  * Rooms ve Building Age alanları metinden sayısala parse edildi (örn. 3+1 → 4).
  * Alan bazlı ek özellikler: log_area, area_per_room, room_bath_ratio, age_bucket.
  * District/Neighborhood için Smoothed Target Encoding (sızıntıyı azaltacak şekilde) üretildi.
  * Varsa lüks özelliklerden luxury_score, ilçeden wealth_score eklendi.

- Seçilen Kritik Özellikler (Feature Selection):
  * Konum (District/Neighborhood), m², oda/banyo, bina yaşı ve türetilen alan oranları.
  * Fiyatı doğrudan/örtük taşıyabilecek kolonlar çıkarıldı (leakage guard).

------------------------------------------------------------------------
4. MODEL MİMARİSİ
------------------------------------------------------------------------
- Kullanılan Algoritma: CatBoostRegressor (log hedef ile)
- Neden bu algoritma?:
  * Kategorik değişkenleri doğal şekilde işleyebilmesi (District/Neighborhood gibi),
  * Tabular veri üzerinde yüksek performans ve stabil öğrenme,
  * Early stopping ile overfit kontrolü yapılabilmesi.
- Elde Edilen Başarı Skoru (RMSE / R-Square):
  * Model eğitim sürecinde train/valid/test ayrımı ve early stopping kullanılmıştır.
  * (Buraya en son çıktınızdaki Test R² ve RMSE değerini yazın.)
    Örn: Test R² ≈ 0.86-0.88, RMSE ≈ 340k-380k TL (veri 2020 ölçeği)

------------------------------------------------------------------------
5. YATIRIM KARAR MANTIĞI 
------------------------------------------------------------------------
Model Fair Value (adil değer) tahmini üretir. İlan fiyatı ile kıyaslanır ve yüzde fark hesaplanır:

Fark(%) = (FairValue - IlanFiyati) / IlanFiyati * 100

Eşik değeri (Threshold) eğitim verisindeki hata dağılımından robust MAD istatistiğiyle otomatik hesaplandı
(örneğin ±8.05%). Böylece eşik rastgele değil, veriye göre kalibre edilmiştir.

- FIRSAT Eşiği:
  Eğer Fark(%) > +Threshold ise → FIRSAT

- PAHALI Eşiği:
  Eğer Fark(%) < -Threshold ise → PAHALI

- Diğer durum:
  |Fark(%)| <= Threshold ise → NORMAL

------------------------------------------------------------------------
6. SİZİ DİĞERLERİNDEN AYIRAN ÖZELLİK 
------------------------------------------------------------------------
- Veriye dayalı, robust (MAD tabanlı) otomatik threshold ile karar bandı oluşturduk.
- Kategorik konum bilgisini güçlendirmek için Smoothed Target Encoding kullandık.
- Streamlit arayüzü ile jüri tek ekranda “Fair Value + Tavsiye” çıktısını anında görebilir.
- Demo senaryosunda aynı ev için farklı ilan fiyatları girilerek FIRSAT/NORMAL/PAHALI davranışı net şekilde gösterilebilir.
