import sqlite3
import pandas as pd
import os

def extract_features():
    print("🧠 SQL Motoru Çalışıyor: Feature Engineering Başladı...")
    
    # Veritabanı yolu
    db_path = os.path.join('data', 'telecom.db')
    conn = sqlite3.connect(db_path)
    
    # --- SENIOR SEVIYE SQL SORGUSU ---
    # Bu sorgu ham veriyi alır, hesaplar ve ML için tek satıra indirger.
    
    query = """
    /* 1. ADIM: Arama İstatistiklerini Hesapla (CTE) */
    WITH CallStats AS (
        SELECT 
            customer_id,
            COUNT(*) as total_calls,
            AVG(duration_minutes) as avg_call_duration,
            SUM(duration_minutes) as total_talk_time,
            /* Son 30 gündeki aktivite (Churn sinyali olabilir) */
            SUM(CASE WHEN call_date >= DATE('now', '-30 days') THEN 1 ELSE 0 END) as calls_last_30_days
        FROM calls
        GROUP BY customer_id
    ),
    
    /* 2. ADIM: Şikayet İstatistiklerini Hesapla (CTE) */
    ComplaintStats AS (
        SELECT 
            customer_id,
            COUNT(*) as total_complaints,
            /* Faturalama ile ilgili şikayeti var mı? (Kritik!) */
            MAX(CASE WHEN topic = 'Billing' THEN 1 ELSE 0 END) as has_billing_issue,
            /* Son 14 günde şikayet etti mi? (Acil Risk) */
            MAX(CASE WHEN complaint_date >= DATE('now', '-14 days') THEN 1 ELSE 0 END) as recent_complaint_flag
        FROM complaints
        GROUP BY customer_id
    )
    
    /* 3. ADIM: Ana Tabloyu Oluştur (Main Join) */
    SELECT 
        c.customer_id,
        c.age,
        c.gender,
        c.contract_type,
        c.monthly_charges,
        c.tenure_months,
        
        /* Arama Özelliklerini Ekle (Boşsa 0 yap) */
        COALESCE(cs.total_calls, 0) as total_calls,
        COALESCE(cs.avg_call_duration, 0) as avg_call_duration,
        COALESCE(cs.calls_last_30_days, 0) as calls_last_30_days,
        
        /* Şikayet Özelliklerini Ekle */
        COALESCE(cps.total_complaints, 0) as total_complaints,
        COALESCE(cps.has_billing_issue, 0) as has_billing_issue,
        COALESCE(cps.recent_complaint_flag, 0) as recent_complaint_flag,
        
        /* Hedef Değişken (Bunu tahmin edeceğiz) */
        c.churn_label
        
    FROM customers c
    LEFT JOIN CallStats cs ON c.customer_id = cs.customer_id
    LEFT JOIN ComplaintStats cps ON c.customer_id = cps.customer_id
    """
    
    # Sorguyu çalıştır ve Pandas DataFrame'e çevir
    df_features = pd.read_sql(query, conn)
    conn.close()
    
    print(f"✅ Özellikler Çıkarıldı! Tablo Boyutu: {df_features.shape}")
    print("   -> Örnek Özellikler: avg_call_duration, has_billing_issue, recent_complaint_flag")
    
    # ML için hazır veriyi kaydet (Intermediate Step)
    df_features.to_csv(os.path.join('data', 'final_features.csv'), index=False)
    print("💾 Veri 'data/final_features.csv' olarak kaydedildi.")
    
    return df_features

if __name__ == "__main__":
    extract_features()