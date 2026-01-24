import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import os

def train_churn_model():
    print("🚀 Model Eğitimi Başlıyor...")

    # 1. VERİYİ YÜKLE
    # Önceki adımda SQL'den çıkarıp kaydettiğimiz dosyayı okuyoruz
    data_path = os.path.join('data', 'final_features.csv')
    
    if not os.path.exists(data_path):
        print("❌ HATA: 'final_features.csv' bulunamadı.")
        print("👉 Lütfen önce 'python3 src/feature_store.py' komutunu çalıştırın!")
        return

    df = pd.read_csv(data_path)
    print(f"📂 Veri Yüklendi: {df.shape[0]} satır, {df.shape[1]} sütun")
    
    # 2. VERİ HAZIRLIĞI
    # Modelin öğrenmemesi gereken (ID) ve hedef (Churn) kolonlarını ayır
    X = df.drop(['churn_label', 'customer_id'], axis=1)
    y = df['churn_label']
    
    # Kategorik değişkenleri belirle (CatBoost bunları çok sever)
    # Veritabanında text olarak tuttuğumuz alanlar:
    categorical_features = ['gender', 'contract_type']
    
    # Train / Test Ayırımı (%80 Eğitim, %20 Test)
    # stratify=y -> Churn oranı iki tarafta da eşit olsun diye
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📊 Eğitim Seti: {len(X_train)} kişi | Test Seti: {len(X_test)} kişi")

    # 3. CATBOOST MODELİNİ KUR
    # auto_class_weights='Balanced': Churn edenler azınlıkta olduğu için onları daha ciddiye al
    model = CatBoostClassifier(
        iterations=500,        # 500 ağaç dik
        depth=6,               # Ağaç derinliği
        learning_rate=0.05,    # Öğrenme hızı
        loss_function='Logloss',
        auto_class_weights='Balanced', 
        verbose=100            # Her 100 adımda bir bilgi ver
    )
    
    # Modeli Eğit
    print("🧠 Model öğreniyor (Bu işlem 5-10 saniye sürebilir)...")
    model.fit(
        X_train, y_train,
        cat_features=categorical_features,
        eval_set=(X_test, y_test),
        early_stopping_rounds=50
    )
    
    # 4. DEĞERLENDİRME (Karne Zamanı)
    print("\n📝 MODEL PERFORMANSI:")
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Detaylı Rapor
    print(classification_report(y_test, y_pred))
    print(f"🌟 ROC-AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 5. MODELİ KAYDET
    # Eğittiğimiz modeli daha sonra Dashboard'da kullanmak için saklıyoruz
    model_path = os.path.join('data', 'churn_model.cbm')
    model.save_model(model_path)
    print(f"💾 Model başarıyla kaydedildi: {model_path}")
    
    # Feature Importance (Hangi özellik daha önemli?)
    print("\n🔍 Müşteriler Neden Gidiyor? (En Önemli 3 Sebep):")
    importance = model.get_feature_importance(prettified=True)
    print(importance.head(3))

if __name__ == "__main__":
    train_churn_model()