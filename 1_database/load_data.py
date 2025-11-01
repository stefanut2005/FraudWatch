# ...existing code...
import os
import time
import tempfile
import shutil
import traceback
import argparse
import pandas as pd
from sqlalchemy import create_engine, text

# Configuration (can be overridden via environment variables)
CSV_FILENAME = 'fraud_transactions.csv'
DB_USER = os.environ.get('DB_USER', os.environ.get('POSTGRES_USER', 'user'))
DB_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get('POSTGRES_PASSWORD', 'pass123'))
DB_HOST = os.environ.get('DB_HOST', 'localhost')
DB_PORT = os.environ.get('DB_PORT', '5432')
DB_NAME = os.environ.get('DB_NAME', os.environ.get('POSTGRES_DB', 'fraud_detection_db'))
TABLE_NAME = os.environ.get('TABLE_NAME', 'transactions')
# FAST_LOAD=1 enables COPY path
FAST_LOAD = os.environ.get('FAST_LOAD', '1') == '1'

# Default test rows
DEFAULT_TEST_ROWS = 10000

def normalize_columns(cols):
    out = []
    for i, c in enumerate(cols):
        if c is None:
            name = f"col_{i+1}"
        else:
            name = str(c).strip().lower().replace(' ', '_')
            if name == "" or name.startswith('"') and name.endswith('"') and len(name.strip('"')) == 0:
                name = f"col_{i+1}"
        out.append(name)
    # ensure uniqueness
    seen = {}
    for idx, n in enumerate(out):
        base = n
        count = seen.get(base, 0)
        if count:
            n = f"{base}_{count}"
            out[idx] = n
        seen[base] = count + 1
    return out

def create_table_schema_from_sample(engine, csv_path, table_name, sample_rows=1000, test_rows=None):
    use_rows = sample_rows
    if test_rows is not None:
        use_rows = min(sample_rows, test_rows)
    df_sample = pd.read_csv(csv_path, nrows=use_rows)
    df_sample.columns = normalize_columns(df_sample.columns)
    df_empty = df_sample.iloc[0:0]
    df_empty.to_sql(table_name, engine, if_exists='replace', index=False)

def copy_csv_to_table(engine, csv_path, table_name, test_rows=None):
    # prepare temporary CSV with normalized header and limited rows if test_rows is set
    with open(csv_path, 'r', encoding='utf-8', errors='ignore') as src:
        header_line = src.readline()
        raw_cols = header_line.strip().split(',')
        cols = normalize_columns(raw_cols)
        with tempfile.NamedTemporaryFile('w+', delete=False, encoding='utf-8') as tmp:
            tmp_name = tmp.name
            tmp.write(','.join(cols) + '\n')
            if test_rows is None:
                shutil.copyfileobj(src, tmp)
            else:
                written = 0
                for line in src:
                    if written >= test_rows:
                        break
                    tmp.write(line)
                    written += 1
    # Ensure table schema matches normalized header: drop & create with TEXT columns
    cols_defs = ', '.join([f'"{c}" TEXT' for c in cols])
    create_sql = f'DROP TABLE IF EXISTS "{table_name}"; CREATE TABLE "{table_name}" ({cols_defs});'
    try:
        with engine.begin() as conn_exec:
            conn_exec.execute(text(create_sql))
    except Exception:
        # if DDL fails, continue and let COPY fail with a helpful traceback
        traceback.print_exc()

    # perform COPY using raw connection and explicit column list
    conn = None
    cur = None
    try:
        conn = engine.raw_connection()
        cur = conn.cursor()
        cols_sql = ', '.join([f'"{c}"' for c in cols])
        sql = f'COPY "{table_name}" ({cols_sql}) FROM STDIN WITH CSV HEADER'
        with open(tmp_name, 'r', encoding='utf-8', errors='ignore') as f:
            cur.copy_expert(sql, f)
        conn.commit()
    finally:
        try:
            if cur:
                cur.close()
        except:
            pass
        try:
            if conn:
                conn.close()
        except:
            pass
        try:
            os.remove(tmp_name)
        except:
            pass

def pandas_chunked_load(engine, csv_path, table_name, chunk_size=5000, test_rows=None):
    start_time = time.time()
    total_rows = 0
    chunk_count = 0
    chunk_reader = pd.read_csv(csv_path, chunksize=chunk_size, iterator=True)
    for i, chunk in enumerate(chunk_reader):
        # apply test_rows limit
        if test_rows is not None:
            remaining = test_rows - total_rows
            if remaining <= 0:
                break
            if len(chunk) > remaining:
                chunk = chunk.iloc[:remaining]
        chunk.columns = normalize_columns(chunk.columns)
        write_mode = 'replace' if i == 0 else 'append'
        chunk.to_sql(table_name, engine, if_exists=write_mode, index=False, method='multi')
        total_rows += len(chunk)
        chunk_count += 1
        elapsed = time.time() - start_time
        rows_per_sec = total_rows / elapsed if elapsed > 0 else 0
        print(f"  ✓ Chunk {chunk_count}: {total_rows:,} rânduri procesate ({rows_per_sec:.0f} rânduri/sec)")
        if test_rows is not None and total_rows >= test_rows:
            break
    elapsed_total = time.time() - start_time
    print(f"\n✅ FINALIZAT: {total_rows:,} rânduri încărcate în '{table_name}'.")
    if elapsed_total > 0:
        print(f"⏱️  Timp total: {elapsed_total:.1f} secunde ({total_rows/elapsed_total:.0f} rânduri/sec)")

def load_data(csv_filename, database_url, table_name, fast_load=True, test_rows=None):
    csv_path = csv_filename

    if not os.path.exists(csv_path):
        print(f"EROARE: Fișierul {csv_filename} nu a fost găsit la calea: {os.path.abspath(csv_path)}")
        return

    print(f"Testează conexiunea la baza de date: {database_url}")
    try:
        engine = create_engine(database_url)
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
        print("Conexiunea la baza de date a fost realizată cu succes.")
    except Exception as e:
        print("EROARE: Conectarea la baza de date a eșuat.")
        print(e)
        return

    print("Începe citirea fișierului CSV...")
    try:
        if fast_load:
            try:
                print("Încerc încărcare rapidă (COPY)...")
                create_table_schema_from_sample(engine, csv_path, table_name, sample_rows=1000, test_rows=test_rows)
                copy_csv_to_table(engine, csv_path, table_name, test_rows=test_rows)
                print(f"\n✅ COPY FINALIZAT: date încărcate în '{table_name}'.")
                return
            except Exception as e:
                print("WARN: COPY a eșuat, se revine la încărcare cu pandas chunked.")
                traceback.print_exc()

        pandas_chunked_load(engine, csv_path, table_name, chunk_size=5000, test_rows=test_rows)
    except Exception as e:
        print(f"\n❌ EROARE la citirea sau scrierea CSV-ului: {e}")
        traceback.print_exc()

def parse_args_and_run():
    parser = argparse.ArgumentParser(description="Load fraud_transactions.csv into Postgres (test mode default: 10k rows).")
    group = parser.add_mutually_exclusive_group()
    group.add_argument('--full', action='store_true', help="Load entire CSV (disable test limit).")
    group.add_argument('--rows', type=int, help="Number of rows to load (overrides default).")
    parser.add_argument('--no-copy', action='store_true', help="Disable COPY fast path and use pandas only.")
    parser.add_argument('--table', type=str, default=TABLE_NAME, help="Target table name.")
    parser.add_argument('--file', type=str, default=CSV_FILENAME, help="CSV file path (defaults to fraud_transactions.csv).")
    args = parser.parse_args()

    if args.full:
        test_rows = None
    elif args.rows is not None:
        test_rows = args.rows
    else:
        test_rows = DEFAULT_TEST_ROWS

    csv_file = args.file
    db_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
    load_data(csv_file, db_url, args.table, fast_load=not args.no_copy and FAST_LOAD, test_rows=test_rows)

if __name__ == "__main__":
    parse_args_and_run()
# ...existing code...