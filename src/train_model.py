import pandas as pd
from catboost import CatBoostClassifier, Pool
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score
import joblib
import os

def train_churn_model():
    print("🚀 Model Eğitimi Başlıyor...")

    # 1. VERİYİ YÜKLE
    data_path = os.path.join('data', 'final_features.csv')
    if not os.path.exists(data_path):
        print("❌ HATA: 'final_features.csv' bulunamadı. Önce 'feature_store.py' çalıştırın!")
        return

    df = pd.read_csv(data_path)
    
    # Gereksiz kolonları çıkar (Müşteri ID modelin işine yaramaz)
    X = df.drop(['churn_label', 'customer_id'], axis=1)
    y = df['churn_label']
    
    # Kategorik değişkenleri belirle (CatBoost'un en sevdiği şey)
    categorical_features = ['gender', 'contract_type']
    
    # 2. TRAIN / TEST AYIRIMI (%80 Eğitim, %20 Test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print(f"📊 Veri Seti: {len(X_train)} Eğitim, {len(X_test)} Test verisi.")

    # 3. CATBOOST MODELİNİ KUR
    # auto_class_weights='Balanced': Churn az olduğu için modele "Azınlığı önemse" diyoruz.
    model = CatBoostClassifier(
        iterations=500, 
        depth=6, 
        learning_rate=0.05,
        loss_function='Logloss',
        auto_class_weights='Balanced', 
        verbose=100  # Her 100 adımda bilgi ver
    )
    
    # Modeli Eğit
    print("🧠 Model öğreniyor...")
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
    
    print(classification_report(y_test, y_pred))
    print(f"🌟 ROC-AUC Skoru: {roc_auc_score(y_test, y_prob):.4f}")
    
    # 5. MODELİ KAYDET
    model_path = os.path.join('data', 'churn_model.cbm')
    model.save_model(model_path)
    print(f"💾 Model kaydedildi: {model_path}")
    
    # Feature Importance (Hangi özellik daha önemli?)
    print("\n🔍 EN ÖNEMLİ 3 SİNYAL (Neden Gidiyorlar?):")
    importance = model.get_feature_importance(prettified=True)
    print(importance.head(3))

if __name__ == "__main__":
    train_churn_model()