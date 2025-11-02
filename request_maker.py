import requests
import json
import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import os
import io
import csv
from sqlalchemy import create_engine

# Server configuration
SERVER_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{SERVER_URL}/predict_fraud"

def generate_random_transaction(save_to_db: bool = True):
    """
    Generate a random transaction in the format expected by the updated MCP server.
    Now optimized for the new Random Forest model with engineered features.
    """
    
    # Generate random dates
    base_date = datetime(2020, 1, 1)  # Updated to match fraud_test.csv timeframe
    trans_date = base_date + timedelta(days=random.randint(0, 1095),  # ~3 years
                                      hours=random.randint(0, 23), 
                                      minutes=random.randint(0, 59))
    
    # Generate birth date (age between 18-80)
    birth_year = trans_date.year - random.randint(18, 80)
    dob = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
    
    # Updated sample data based on fraud_test.csv analysis
    categories = ['grocery_pos', 'entertainment', 'gas_transport', 'misc_net', 'shopping_net', 
                 'kids_pets', 'food_dining', 'personal_care', 'health_fitness', 'travel',
                 'shopping_pos', 'home', 'misc_pos']
    
    jobs = ['Mechanical engineer', 'Teacher, primary school', 'Solicitor', 'Psychologist, clinical',
           'Sales representative', 'Software engineer', 'Marketing manager', 'Nurse', 'Accountant',
           'Project manager', 'Doctor', 'Analyst', 'Designer', 'Consultant', 'Developer']
    
    first_names = ['Jennifer', 'Michael', 'Joshua', 'Michelle', 'David', 'Lisa', 'Christopher',
                  'Daniel', 'Sarah', 'Matthew', 'Ashley', 'Anthony', 'Mark', 'Donald', 'Steven',
                  'Paul', 'Andrew', 'Kenneth', 'William', 'John', 'Robert', 'James']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson']
    
    genders = ['M', 'F']
    
    streets = ['Main St', 'Oak Ave', 'Pine Rd', 'Elm St', 'First Ave', 'Second St', 'Park Ave',
              'Washington St', 'Lincoln Ave', 'Broadway', 'Maple Dr', 'Cedar Ln', 'Sunset Blvd']
    
    # Realistic transaction amounts with some fraud-like patterns
    if random.random() < 0.05:  # 5% chance of high-value transaction
        amt = round(random.uniform(500.0, 5000.0), 2)
    else:
        amt = round(np.random.exponential(50), 2)  # Most transactions are small
        
    # US geographic coordinates
    lat = round(random.uniform(25.0, 49.0), 6)
    long = round(random.uniform(-125.0, -66.0), 6)
    
    transaction_data = {
        'trans_date_trans_time': trans_date.strftime('%d/%m/%Y %H:%M'),
        'dob': dob.strftime('%d/%m/%Y'),
        'category': random.choice(categories),
        'job': random.choice(jobs),
        'first': random.choice(first_names),
        'last': random.choice(last_names),
        'gender': random.choice(genders),
        'street': random.choice(streets),
        'amt': amt,
        'lat': lat,
        'long': long,
        'city_pop': random.randint(1000, 8000000),  # Realistic city population range
        'merch_lat': round(lat + random.uniform(-5.0, 5.0), 6),  # Merchant near transaction
        'merch_long': round(long + random.uniform(-5.0, 5.0), 6),
        'cc_num': random.randint(1000000000000000, 9999999999999999),
        'zip': random.randint(10000, 99999),
        'unix_time': int(trans_date.timestamp()),
        'trans_num': f"TXN_{random.randint(10000, 99999)}",
        'merchant': f"fraud_{random.choice(['Kirlin and Sons', 'Rippin, Kub and Mann', 'Weber, Collins and Welch', 'Johns Group', 'Mueller-Moore'])}",
        'state': random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'SC']),
        'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
                              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Columbia']),
        'id': random.randint(0, 999999)  # Match fraud_test.csv ID format
    }
    
    # Optionally persist the generated transaction into the database
    if save_to_db:
        try:
            save_transaction_to_db(transaction_data)
        except Exception as e:
            print(f"Warning: failed to save generated transaction to DB: {e}")

    return transaction_data


def save_transaction_to_db(transaction: dict, table_name: str = 'random_tranzactions') -> bool:
    """Insert a single transaction dict into the Postgres table `random_tranzactions`.

    Reads DB connection settings from environment variables (same defaults as other scripts).
    Returns True on success, False on failure.
    """
    DB_USER = os.environ.get('DB_USER', os.environ.get('POSTGRES_USER', 'user'))
    DB_PASSWORD = os.environ.get('DB_PASSWORD', os.environ.get('POSTGRES_PASSWORD', 'pass123'))
    DB_HOST = os.environ.get('DB_HOST', 'localhost')
    DB_PORT = os.environ.get('DB_PORT', '5432')
    DB_NAME = os.environ.get('DB_NAME', os.environ.get('POSTGRES_DB', 'fraud_detection_db'))

    DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

    try:
        engine = create_engine(DATABASE_URL)

        # Try fast COPY upload using raw DBAPI connection (psycopg2)
        try:
            conn = engine.raw_connection()
            cur = conn.cursor()

            # Prepare CSV in-memory with header and single row
            columns = list(transaction.keys())
            sio = io.StringIO()
            writer = csv.writer(sio)
            # write header
            writer.writerow(columns)
            # write row values in the same order
            row = [transaction.get(col, None) for col in columns]
            # Convert non-string types (e.g., dicts) to JSON strings
            def normalize_val(v):
                if v is None:
                    return ''
                if isinstance(v, (dict, list)):
                    return json.dumps(v)
                return v
            row = [normalize_val(v) for v in row]
            writer.writerow(row)
            sio.seek(0)

            cols_sql = ', '.join([f'"{c}"' for c in columns])
            sql = f"COPY {table_name} ({cols_sql}) FROM STDIN WITH CSV HEADER"
            cur.copy_expert(sql, sio)
            conn.commit()
            try:
                cur.close()
            except Exception:
                pass
            try:
                conn.close()
            except Exception:
                pass
            try:
                engine.dispose()
            except Exception:
                pass
            return True
        except Exception as copy_exc:
            # Fallback to pandas.to_sql if COPY is not available or fails (table may not exist)
            print(f"COPY upload failed, falling back to pandas.to_sql: {copy_exc}")
            try:
                df = pd.DataFrame([transaction])
                df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]
                df.to_sql(table_name, engine, if_exists='append', index=False)
                try:
                    engine.dispose()
                except Exception:
                    pass
                return True
            except Exception as e:
                print(f"Error saving transaction to DB with pandas.to_sql ({table_name}): {e}")
                return False

    except Exception as e:
        print(f"Error saving transaction to DB ({table_name}): {e}")
        return False

def test_predict_fraud_endpoint(num_requests=5):
    """
    Test the predict_fraud endpoint with random transaction data.
    
    Args:
        num_requests (int): Number of test requests to make
    """
    print(f"Testing predict_fraud endpoint at {PREDICT_ENDPOINT}")
    print(f"Sending {num_requests} random transaction requests...\n")
    
    results = []
    
    for i in range(num_requests):
        # Generate random transaction data
        transaction_data = generate_random_transaction()
        
        print(f"Request {i+1}:")
        print(f"Transaction Data: {json.dumps(transaction_data, indent=2)}")
        
        try:
            # Send POST request to the predict_fraud endpoint
            response = requests.post(
                PREDICT_ENDPOINT,
                json=transaction_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result}")
                results.append({
                    'request_id': i+1,
                    'transaction_amount': transaction_data['amt'],
                    'category': transaction_data['category'],
                    'fraud_detected': result.get('fraud_detected', False),
                    'confidence': result.get('confidence', 0.0),
                    'status': 'success'
                })
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Error: {error_msg}")
                results.append({
                    'request_id': i+1,
                    'status': 'error',
                    'error': error_msg
                })
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"❌ Request Error: {error_msg}")
            results.append({
                'request_id': i+1,
                'status': 'error',
                'error': error_msg
            })
        
        print("-" * 50)
    
    # Summary
    successful_requests = [r for r in results if r['status'] == 'success']
    fraud_detected = [r for r in successful_requests if r.get('fraud_detected', False)]
    
    print("\n📊 SUMMARY:")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {num_requests - len(successful_requests)}")
    if successful_requests:
        print(f"Fraud detected in: {len(fraud_detected)} transactions")
        print(f"Fraud detection rate: {len(fraud_detected)/len(successful_requests)*100:.1f}%")
    
    return results

def test_server_connection():
    """Test if the server is running and accessible."""
    try:
        response = requests.get(SERVER_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            return True
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the MCP server is running on http://127.0.0.1:8000")
        return False

def load_csv_and_test(csv_file_path, num_samples=5):
    """
    Load transactions from CSV file and test with random samples.
    Updated to work with fraud_test.csv format.
    
    Args:
        csv_file_path (str): Path to the CSV file
        num_samples (int): Number of random samples to test
    """
    try:
        print(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} transactions from CSV")
        
        if len(df) == 0:
            print("❌ CSV file is empty")
            return
        
        # Get random samples
        sample_df = df.sample(n=min(num_samples, len(df)))
        
        print(f"\nTesting with {len(sample_df)} random samples from CSV...")
        
        results = []
        for idx, row in sample_df.iterrows():
            print(f"\nRequest {len(results)+1}:")
            
            # Convert CSV row to expected format - fraud_test.csv has the right structure
            transaction_data = {
                'trans_date_trans_time': row.get('trans_date_trans_time', '01/01/2020 12:00'),
                'dob': row.get('dob', '01/01/1980'),
                'category': row.get('category', 'misc_net'),
                'job': row.get('job', 'Engineer'),
                'first': row.get('first', 'John'),
                'last': row.get('last', 'Doe'),
                'gender': row.get('gender', 'M'),
                'street': row.get('street', 'Main St'),
                'amt': float(row.get('amt', 100.0)),
                'lat': float(row.get('lat', 40.7128)),
                'long': float(row.get('long', -74.0060)),
                'city_pop': int(row.get('city_pop', 8000000)),
                'merch_lat': float(row.get('merch_lat', 40.7128)),
                'merch_long': float(row.get('merch_long', -74.0060)),
                'cc_num': int(row.get('cc_num', 1234567890123456)),
                'zip': int(row.get('zip', 10001)),
                'unix_time': int(row.get('unix_time', 1577836800)),
                'trans_num': row.get('trans_num', 'TXN_12345'),
                'merchant': row.get('merchant', 'Test Merchant'),
                'state': row.get('state', 'NY'),
                'city': row.get('city', 'New York'),
                'id': int(row.get('id', 0))
            }
            
            actual_fraud = row.get('is_fraud', 0)
            print(f"Testing transaction ID {transaction_data['id']} - Amount: ${transaction_data['amt']:.2f}")
            if actual_fraud == 1:
                print(f"⚠️  This is an ACTUAL FRAUD case from the dataset!")
            
            try:
                response = requests.post(
                    PREDICT_ENDPOINT,
                    json=transaction_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_fraud = result.get('fraud_detected', False)
                    confidence = result.get('confidence', 0.0)
                    
                    # Check if prediction matches actual
                    is_correct = (predicted_fraud and actual_fraud == 1) or (not predicted_fraud and actual_fraud == 0)
                    
                    print(f"✅ Response: {result}")
                    if actual_fraud == 1:
                        print(f"🎯 Fraud Detection: {'✅ CAUGHT' if predicted_fraud else '❌ MISSED'}")
                    
                    results.append({
                        'original_row': idx,
                        'amount': transaction_data['amt'],
                        'actual_fraud': actual_fraud,
                        'predicted_fraud': predicted_fraud,
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'status': 'success'
                    })
                else:
                    print(f"❌ Error: HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request Error: {str(e)}")
        
        # Summary for CSV testing
        if results:
            successful_requests = [r for r in results if r['status'] == 'success']
            fraud_cases = [r for r in successful_requests if r['actual_fraud'] == 1]
            correctly_identified_fraud = [r for r in fraud_cases if r['predicted_fraud']]
            
            print(f"\n📊 CSV TESTING SUMMARY:")
            print(f"Total requests: {len(results)}")
            print(f"Actual fraud cases in sample: {len(fraud_cases)}")
            if fraud_cases:
                print(f"Fraud cases correctly identified: {len(correctly_identified_fraud)}/{len(fraud_cases)} ({len(correctly_identified_fraud)/len(fraud_cases)*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

if __name__ == "__main__":
    import sys
    
    print("🚀 Fraud Detection API Tester")
    print("=" * 50)
    
    # Test server connection first
    if not test_server_connection():
        print("\n💡 To start the server, run:")
        print("cd 3_ai_agent && python mcp_server.py")
        exit(1)
    
    # Check for command line arguments
    if len(sys.argv) > 1:
        if "--random" in sys.argv:
            # Extract number of samples
            num_requests = 5
            for i, arg in enumerate(sys.argv):
                if arg == "--num-samples" and i + 1 < len(sys.argv):
                    try:
                        num_requests = int(sys.argv[i + 1])
                    except ValueError:
                        pass
            test_predict_fraud_endpoint(num_requests)
        elif "--test-csv" in sys.argv:
            # Extract CSV path and number of samples
            csv_path = "fraud_test.csv"
            num_samples = 5
            for i, arg in enumerate(sys.argv):
                if arg == "--csv-path" and i + 1 < len(sys.argv):
                    csv_path = sys.argv[i + 1]
                elif arg == "--num-samples" and i + 1 < len(sys.argv):
                    try:
                        num_samples = int(sys.argv[i + 1])
                    except ValueError:
                        pass
            load_csv_and_test(csv_path, num_samples)
        else:
            print("Usage: python request_maker.py [--random|--test-csv] [--num-samples N] [--csv-path PATH]")
            exit(1)
    else:
        # Interactive mode
        print("\nChoose testing mode:")
        print("1. Test with random generated data")
        print("2. Test with data from CSV file")
        
        choice = input("Enter your choice (1 or 2): ").strip()
        
        if choice == "1":
            num_requests = input("Enter number of test requests (default 5): ").strip()
            num_requests = int(num_requests) if num_requests.isdigit() else 5
            test_predict_fraud_endpoint(num_requests)
        
        elif choice == "2":
            csv_path = input("Enter CSV file path (or press Enter for default): ").strip()
            if not csv_path:
                csv_path = "fraud_test.csv"  # Updated to use the new dataset
            
            num_samples = input("Enter number of samples to test (default 5): ").strip()
            num_samples = int(num_samples) if num_samples.isdigit() else 5
            
            load_csv_and_test(csv_path, num_samples)
        
        else:
            print("Invalid choice. Running with random data...")
            test_predict_fraud_endpoint(5)

import requests
import json
import random
import numpy as np
from datetime import datetime, timedelta
import pandas as pd

# Server configuration
SERVER_URL = "http://127.0.0.1:8000"
PREDICT_ENDPOINT = f"{SERVER_URL}/predict_fraud"

def generate_random_transaction():
    """
    Generate a random transaction in the format expected by the updated MCP server.
    Now optimized for the new Random Forest model with engineered features.
    """
    
    # Generate random dates
    base_date = datetime(2020, 1, 1)  # Updated to match fraud_test.csv timeframe
    trans_date = base_date + timedelta(days=random.randint(0, 1095),  # ~3 years
                                      hours=random.randint(0, 23), 
                                      minutes=random.randint(0, 59))
    
    # Generate birth date (age between 18-80)
    birth_year = trans_date.year - random.randint(18, 80)
    dob = datetime(birth_year, random.randint(1, 12), random.randint(1, 28))
    
    # Updated sample data based on fraud_test.csv analysis
    categories = ['grocery_pos', 'entertainment', 'gas_transport', 'misc_net', 'shopping_net', 
                 'kids_pets', 'food_dining', 'personal_care', 'health_fitness', 'travel',
                 'shopping_pos', 'home', 'misc_pos']
    
    jobs = ['Mechanical engineer', 'Teacher, primary school', 'Solicitor', 'Psychologist, clinical',
           'Sales representative', 'Software engineer', 'Marketing manager', 'Nurse', 'Accountant',
           'Project manager', 'Doctor', 'Analyst', 'Designer', 'Consultant', 'Developer']
    
    first_names = ['Jennifer', 'Michael', 'Joshua', 'Michelle', 'David', 'Lisa', 'Christopher',
                  'Daniel', 'Sarah', 'Matthew', 'Ashley', 'Anthony', 'Mark', 'Donald', 'Steven',
                  'Paul', 'Andrew', 'Kenneth', 'William', 'John', 'Robert', 'James']
    
    last_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones', 'Garcia', 'Miller', 'Davis',
                 'Rodriguez', 'Martinez', 'Hernandez', 'Lopez', 'Gonzalez', 'Wilson', 'Anderson',
                 'Thomas', 'Taylor', 'Moore', 'Jackson', 'Martin', 'Lee', 'Thompson']
    
    genders = ['M', 'F']
    
    streets = ['Main St', 'Oak Ave', 'Pine Rd', 'Elm St', 'First Ave', 'Second St', 'Park Ave',
              'Washington St', 'Lincoln Ave', 'Broadway', 'Maple Dr', 'Cedar Ln', 'Sunset Blvd']
    
    # Realistic transaction amounts with some fraud-like patterns
    if random.random() < 0.05:  # 5% chance of high-value transaction
        amt = round(random.uniform(500.0, 5000.0), 2)
    else:
        amt = round(np.random.exponential(50), 2)  # Most transactions are small
        
    # US geographic coordinates
    lat = round(random.uniform(25.0, 49.0), 6)
    long = round(random.uniform(-125.0, -66.0), 6)
    
    transaction_data = {
        'trans_date_trans_time': trans_date.strftime('%d/%m/%Y %H:%M'),
        'dob': dob.strftime('%d/%m/%Y'),
        'category': random.choice(categories),
        'job': random.choice(jobs),
        'first': random.choice(first_names),
        'last': random.choice(last_names),
        'gender': random.choice(genders),
        'street': random.choice(streets),
        'amt': amt,
        'lat': lat,
        'long': long,
        'city_pop': random.randint(1000, 8000000),  # Realistic city population range
        'merch_lat': round(lat + random.uniform(-5.0, 5.0), 6),  # Merchant near transaction
        'merch_long': round(long + random.uniform(-5.0, 5.0), 6),
        'cc_num': random.randint(1000000000000000, 9999999999999999),
        'zip': random.randint(10000, 99999),
        'unix_time': int(trans_date.timestamp()),
        'trans_num': f"TXN_{random.randint(10000, 99999)}",
        'merchant': f"fraud_{random.choice(['Kirlin and Sons', 'Rippin, Kub and Mann', 'Weber, Collins and Welch', 'Johns Group', 'Mueller-Moore'])}",
        'state': random.choice(['CA', 'NY', 'TX', 'FL', 'IL', 'PA', 'OH', 'GA', 'NC', 'MI', 'SC']),
        'city': random.choice(['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix', 'Philadelphia', 
                              'San Antonio', 'San Diego', 'Dallas', 'San Jose', 'Columbia']),
        'id': random.randint(0, 999999)  # Match fraud_test.csv ID format
    }
    
    return transaction_data

def test_predict_fraud_endpoint(num_requests=5):
    """
    Test the predict_fraud endpoint with random transaction data.
    
    Args:
        num_requests (int): Number of test requests to make
    """
    print(f"Testing predict_fraud endpoint at {PREDICT_ENDPOINT}")
    print(f"Sending {num_requests} random transaction requests...\n")
    
    results = []
    
    for i in range(num_requests):
        # Generate random transaction data
        transaction_data = generate_random_transaction()
        
        print(f"Request {i+1}:")
        print(f"Transaction Data: {json.dumps(transaction_data, indent=2)}")
        
        try:
            # Send POST request to the predict_fraud endpoint
            response = requests.post(
                PREDICT_ENDPOINT,
                json=transaction_data,
                headers={'Content-Type': 'application/json'},
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Response: {result}")
                results.append({
                    'request_id': i+1,
                    'transaction_amount': transaction_data['amt'],
                    'category': transaction_data['category'],
                    'fraud_detected': result.get('fraud_detected', False),
                    'confidence': result.get('confidence', 0.0),
                    'status': 'success'
                })
            else:
                error_msg = f"HTTP {response.status_code}: {response.text}"
                print(f"❌ Error: {error_msg}")
                results.append({
                    'request_id': i+1,
                    'status': 'error',
                    'error': error_msg
                })
                
        except requests.exceptions.RequestException as e:
            error_msg = f"Request failed: {str(e)}"
            print(f"❌ Request Error: {error_msg}")
            results.append({
                'request_id': i+1,
                'status': 'error',
                'error': error_msg
            })
        
        print("-" * 50)
    
    # Summary
    successful_requests = [r for r in results if r['status'] == 'success']
    fraud_detected = [r for r in successful_requests if r.get('fraud_detected', False)]
    
    print("\n📊 SUMMARY:")
    print(f"Total requests: {num_requests}")
    print(f"Successful requests: {len(successful_requests)}")
    print(f"Failed requests: {num_requests - len(successful_requests)}")
    if successful_requests:
        print(f"Fraud detected in: {len(fraud_detected)} transactions")
        print(f"Fraud detection rate: {len(fraud_detected)/len(successful_requests)*100:.1f}%")
    
    return results

def test_server_connection():
    """Test if the server is running and accessible."""
    try:
        response = requests.get(SERVER_URL, timeout=5)
        if response.status_code == 200:
            print("✅ Server is running and accessible")
            return True
        else:
            print(f"❌ Server responded with status code: {response.status_code}")
            return False
    except requests.exceptions.RequestException as e:
        print(f"❌ Cannot connect to server: {e}")
        print("Make sure the MCP server is running on http://127.0.0.1:8000")
        return False

def load_csv_and_test(csv_file_path, num_samples=5):
    """
    Load transactions from CSV file and test with random samples.
    Updated to work with fraud_test.csv format.
    
    Args:
        csv_file_path (str): Path to the CSV file
        num_samples (int): Number of random samples to test
    """
    try:
        print(f"Loading data from {csv_file_path}...")
        df = pd.read_csv(csv_file_path)
        print(f"Loaded {len(df)} transactions from CSV")
        
        if len(df) == 0:
            print("❌ CSV file is empty")
            return
        
        # Get random samples
        sample_df = df.sample(n=min(num_samples, len(df)))
        
        print(f"\nTesting with {len(sample_df)} random samples from CSV...")
        
        results = []
        for idx, row in sample_df.iterrows():
            print(f"\nRequest {len(results)+1}:")
            
            # Convert CSV row to expected format - fraud_test.csv has the right structure
            transaction_data = {
                'trans_date_trans_time': row.get('trans_date_trans_time', '01/01/2020 12:00'),
                'dob': row.get('dob', '01/01/1980'),
                'category': row.get('category', 'misc_net'),
                'job': row.get('job', 'Engineer'),
                'first': row.get('first', 'John'),
                'last': row.get('last', 'Doe'),
                'gender': row.get('gender', 'M'),
                'street': row.get('street', 'Main St'),
                'amt': float(row.get('amt', 100.0)),
                'lat': float(row.get('lat', 40.7128)),
                'long': float(row.get('long', -74.0060)),
                'city_pop': int(row.get('city_pop', 8000000)),
                'merch_lat': float(row.get('merch_lat', 40.7128)),
                'merch_long': float(row.get('merch_long', -74.0060)),
                'cc_num': int(row.get('cc_num', 1234567890123456)),
                'zip': int(row.get('zip', 10001)),
                'unix_time': int(row.get('unix_time', 1577836800)),
                'trans_num': row.get('trans_num', 'TXN_12345'),
                'merchant': row.get('merchant', 'Test Merchant'),
                'state': row.get('state', 'NY'),
                'city': row.get('city', 'New York'),
                'id': int(row.get('id', 0))
            }
            
            actual_fraud = row.get('is_fraud', 0)
            print(f"Testing transaction ID {transaction_data['id']} - Amount: ${transaction_data['amt']:.2f}")
            if actual_fraud == 1:
                print(f"⚠️  This is an ACTUAL FRAUD case from the dataset!")
            
            try:
                response = requests.post(
                    PREDICT_ENDPOINT,
                    json=transaction_data,
                    headers={'Content-Type': 'application/json'},
                    timeout=30
                )
                
                if response.status_code == 200:
                    result = response.json()
                    predicted_fraud = result.get('fraud_detected', False)
                    confidence = result.get('confidence', 0.0)
                    
                    # Check if prediction matches actual
                    is_correct = (predicted_fraud and actual_fraud == 1) or (not predicted_fraud and actual_fraud == 0)
                    
                    print(f"✅ Response: {result}")
                    if actual_fraud == 1:
                        print(f"🎯 Fraud Detection: {'✅ CAUGHT' if predicted_fraud else '❌ MISSED'}")
                    
                    results.append({
                        'original_row': idx,
                        'amount': transaction_data['amt'],
                        'actual_fraud': actual_fraud,
                        'predicted_fraud': predicted_fraud,
                        'confidence': confidence,
                        'is_correct': is_correct,
                        'status': 'success'
                    })
                else:
                    print(f"❌ Error: HTTP {response.status_code}: {response.text}")
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request Error: {str(e)}")
        
        # Summary for CSV testing
        if results:
            successful_requests = [r for r in results if r['status'] == 'success']
            fraud_cases = [r for r in successful_requests if r['actual_fraud'] == 1]
            correctly_identified_fraud = [r for r in fraud_cases if r['predicted_fraud']]
            
            print(f"\n📊 CSV TESTING SUMMARY:")
            print(f"Total requests: {len(results)}")
            print(f"Actual fraud cases in sample: {len(fraud_cases)}")
            if fraud_cases:
                print(f"Fraud cases correctly identified: {len(correctly_identified_fraud)}/{len(fraud_cases)} ({len(correctly_identified_fraud)/len(fraud_cases)*100:.1f}%)")
        
        return results
        
    except Exception as e:
        print(f"❌ Error loading CSV: {e}")
        return []

if __name__ == "__main__":
    print("🚀 Fraud Detection API Tester")
    print("=" * 50)
    
    # Test server connection first
    if not test_server_connection():
        print("\n💡 To start the server, run:")
        print("cd 3_ai_agent && python mcp_server.py")
        exit(1)
    
    print("\nChoose testing mode:")
    print("1. Test with random generated data")
    print("2. Test with data from CSV file")
    
    choice = input("Enter your choice (1 or 2): ").strip()
    
    if choice == "1":
        num_requests = input("Enter number of test requests (default 5): ").strip()
        num_requests = int(num_requests) if num_requests.isdigit() else 5
        test_predict_fraud_endpoint(num_requests)
    
    elif choice == "2":
        csv_path = input("Enter CSV file path (or press Enter for default): ").strip()
        if not csv_path:
            csv_path = "fraud_test.csv"  # Updated to use the new dataset
        
        num_samples = input("Enter number of samples to test (default 5): ").strip()
        num_samples = int(num_samples) if num_samples.isdigit() else 5
        
        load_csv_and_test(csv_path, num_samples)
    
    else:
        print("Invalid choice. Running with random data...")
        test_predict_fraud_endpoint(5)
