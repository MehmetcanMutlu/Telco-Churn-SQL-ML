import pandas as pd
import numpy as np
import sqlite3
import random
import os
from datetime import datetime, timedelta

def create_telecom_db():
    print("🏗️  Telekom Veritabanı İnşa Ediliyor...")
    
    # Klasör kontrolü (data klasörü yoksa oluştur)
    if not os.path.exists('data'):
        os.makedirs('data')

    # 1. MÜŞTERİLERİ OLUŞTUR (Customers Table)
    num_customers = 1000
    customer_ids = [1000 + i for i in range(num_customers)]
    
    data_customers = {
        'customer_id': customer_ids,
        'age': np.random.randint(18, 75, num_customers),
        'gender': np.random.choice(['M', 'F'], num_customers),
        'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year'], num_customers, p=[0.5, 0.3, 0.2]),
        'monthly_charges': np.round(np.random.uniform(30, 120, num_customers), 2),
        'tenure_months': np.random.randint(1, 72, num_customers),
        'churn_label': np.random.choice([0, 1], num_customers, p=[0.75, 0.25]) 
    }
    df_customers = pd.DataFrame(data_customers)
    
    # 2. ARAMA KAYITLARI (Calls)
    print("📞  Arama kayıtları simüle ediliyor...")
    calls_data = []
    start_date = datetime.now() - timedelta(days=180)
    
    for _ in range(20000): 
        cid = random.choice(customer_ids)
        is_churn = df_customers.loc[df_customers['customer_id'] == cid, 'churn_label'].values[0]
        call_date = start_date + timedelta(days=random.randint(0, 180))
        duration = np.random.exponential(5) + 1 
        
        if is_churn == 1 and (datetime.now() - call_date).days < 30:
            if random.random() > 0.3: continue
            duration = duration * 0.5 
            
        calls_data.append([cid, call_date.date(), round(duration, 2)])
        
    df_calls = pd.DataFrame(calls_data, columns=['customer_id', 'call_date', 'duration_minutes'])
    
    # 3. ŞİKAYET KAYITLARI (Complaints)
    print("😡  Şikayet verileri ekleniyor...")
    complaints_data = []
    for _ in range(1500):
        cid = random.choice(customer_ids)
        is_churn = df_customers.loc[df_customers['customer_id'] == cid, 'churn_label'].values[0]
        comp_date = start_date + timedelta(days=random.randint(0, 180))
        weight = 0.8 if is_churn == 1 else 0.2
        if random.random() < weight:
            topic = random.choice(['Billing', 'Network', 'Service', 'Data'])
            complaints_data.append([cid, comp_date.date(), topic])
            
    df_complaints = pd.DataFrame(complaints_data, columns=['customer_id', 'complaint_date', 'topic'])

    # 4. KAYDET (data/telecom.db içine)
    db_path = os.path.join('data', 'telecom.db')
    conn = sqlite3.connect(db_path)
    
    df_customers.to_sql('customers', conn, if_exists='replace', index=False)
    df_calls.to_sql('calls', conn, if_exists='replace', index=False)
    df_complaints.to_sql('complaints', conn, if_exists='replace', index=False)
    conn.close()
    
    print(f"✅ VERİTABANI HAZIR! '{db_path}' oluşturuldu.")

if __name__ == "__main__":
    create_telecom_db()