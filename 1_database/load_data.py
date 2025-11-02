import pandas as pd
from sqlalchemy import create_engine, text
import time
import os
import sys
import traceback
import tempfile
import shutil

# Configuration (can be overridden via environment variables)
CSV_FILENAME = 'fraud_transactions.csv'
DB_USER = os.environ.get('DB_USER', os.environ.get('POSTGRES_USER', 'user'))
DB_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get('POSTGRES_PASSWORD', 'pass123'))
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', os.environ.get('POSTGRES_DB', 'fraud_detection_db'))
TABLE_NAME = os.environ.get('TABLE_NAME', 'transactions')
# Set FAST_LOAD=0 to force pandas chunked load
FAST_LOAD = os.environ.get('FAST_LOAD', '1') == '1'

DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

def normalize_columns(cols):
    return [col.strip().lower().replace(' ', '_') for col in cols]

# --- Funcțiile de mai jos nu mai sunt folosite de `load_data`, 
# --- dar sunt păstrate dacă vrei să revii la încărcarea completă.

def create_table_schema_from_sample(engine, csv_path, table_name, sample_rows=1000):
    df_sample = pd.read_csv(csv_path, nrows=sample_rows)
    df_sample.columns = normalize_columns(df_sample.columns)
    df_empty = df_sample.iloc[0:0]
    df_empty.to_sql(table_name, engine, if_exists='replace', index=False)

def copy_csv_to_table(engine, csv_path, table_name):
    # prepare temporary CSV with normalized header
    with open(csv_path, 'r', encoding='utf-8') as src:
        header = src.readline()
        cols = normalize_columns(header.strip().split(','))
        with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tmp:
            tmp_name = tmp.name
            tmp.write(','.join(cols) + '\n')
            shutil.copyfileobj(src, tmp)
    # perform COPY using raw connection
    try:
        conn = engine.raw_connection()
        cur = conn.cursor()
        with open(tmp_name, 'r', encoding='utf-8') as f:
            cols_sql = ', '.join([f'"{c}"' for c in cols])
            sql = f"COPY {table_name} ({cols_sql}) FROM STDIN WITH CSV HEADER"
            cur.copy_expert(sql, f)
        conn.commit()
    finally:
        try:
            cur.close()
        except:
            pass
        try:
            conn.close()
        except:
            pass
        try:
            os.remove(tmp_name)
        except:
            pass

def pandas_chunked_load(engine, csv_path, table_name, chunk_size=100000):
    start_time = time.time()
    total_rows = 0
    chunk_count = 0
    chunk_reader = pd.read_csv(csv_path, chunksize=chunk_size)
    for i, chunk in enumerate(chunk_reader):
        chunk.columns = normalize_columns(chunk.columns)
        write_mode = 'replace' if i == 0 else 'append'
        chunk.to_sql(table_name, engine, if_exists=write_mode, index=False, method='multi')
        total_rows += len(chunk)
        chunk_count += 1
        elapsed = time.time() - start_time
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
        print(f"  ✓ Chunk {chunk_count}: {total_rows:,} rânduri procesate ({rows_per_sec:.0f} rânduri/sec)")
    elapsed_total = time.time() - start_time
    print(f"\n✅ FINALIZAT: {total_rows:,} rânduri încărcate în '{table_name}'.")
    print(f"⏱️  Timp total: {elapsed_total:.1f} secunde ({total_rows/elapsed_total:.0f} rânduri/sec)")

# --- AICI ESTE MODIFICAREA ---
def load_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(script_dir, CSV_FILENAME)

    if not os.path.exists(csv_path):
        print(f"EROARE: Fișierul {CSV_FILENAME} nu a fost găsit la calea: {csv_path}")
        print("Descarcă manual 'fraud_transactions.csv' de pe Kaggle și plasează-l în directorul '1_database/'.")
        return

    print(f"Am găsit fișierul {csv_path}.")
    print(f"Încercare de conectare la baza de date: postgresql://{DB_USER}:****@{DB_HOST}:{DB_PORT}/{DB_NAME}")
    try:
        engine = create_engine(DATABASE_URL)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Conexiunea la baza de date a fost realizată cu succes.")
    except Exception as e:
        print("EROARE: Conectarea la baza de date a eșuat.")
        print(e)
        print("Asigură-te că ai rulat 'docker compose up -d' sau 'docker-compose up -d'.")
        return

    # Setează limita de rânduri
    row_limit = 100000

    print(f"Începe citirea primelor {row_limit} rânduri din fișierul CSV...")
    try:
        start_time = time.time()
        
        # 1. Citim doar primele 10.000 de rânduri
        df = pd.read_csv(csv_path, nrows=row_limit)
        
        # 2. Normalizăm coloanele
        df.columns = normalize_columns(df.columns)
        
        # 3. Scriem în baza de date, înlocuind tabelul existent
        print(f"Se scriu {len(df)} rânduri în tabelul '{TABLE_NAME}'...")
        df.to_sql(TABLE_NAME, engine, if_exists='replace', index=False, method='multi')
        
        elapsed = time.time() - start_time
        print(f"\n✅ FINALIZAT: {len(df):,} rânduri încărcate în '{TABLE_NAME}'.")
        print(f"⏱️  Timp total: {elapsed:.1f} secunde.")

    except Exception as e:
        print(f"\n❌ EROARE la citirea sau scrierea CSV-ului: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    load_data()