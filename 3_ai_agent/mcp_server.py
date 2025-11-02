import uvicorn
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
import sys
import os
from tensorflow import keras

#model = keras.models.load_model("../pkl-files/neural_network_fraud_model.h5")
#print("===================================")
#print(model.summary())

# Add path to request_maker.py
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, parent_dir)
try:
    from request_maker import generate_random_transaction
    print(f"✅ Successfully imported generate_random_transaction from {parent_dir}/request_maker.py")
except ImportError as e:
    print(f"❌ Error importing generate_random_transaction: {e}")
    print(f"   Parent dir: {parent_dir}")
    print(f"   Sys.path: {sys.path}")
    raise

# --- Configurare Conexiune Bază de Date ---
# Setările trebuie să se potrivească cu fișierele din 1_database
DB_USER = 'user'
DB_PASSWORD = 'pass123'
DB_HOST = 'localhost' # Se conectează la containerul Docker
DB_PORT = '5432'
DB_NAME = 'fraud_detection_db'
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Inițializare ---
app = FastAPI()
try:
    # Load the neural network model
    print("Loading neural network model...")
    neural_network_model = tf.keras.models.load_model('../pkl-files/neural_network_fraud_model.h5')
    print("✅ Neural network model loaded successfully")

    # Load the feature scaler
    with open('../pkl-files/nn_scaler.pkl', 'rb') as scaler_file:
        feature_scaler = pickle.load(scaler_file)
    print("✅ Feature scaler loaded successfully")

    # Load the optimal threshold for neural network
    with open('../pkl-files/nn_optimal_threshold.pkl', 'rb') as threshold_file:
        threshold_data = pickle.load(threshold_file)
        optimal_threshold = threshold_data['optimal_threshold'] if isinstance(threshold_data, dict) else threshold_data
    print(f"✅ Optimal threshold loaded: {optimal_threshold}")

    # Load the label encoders (still needed for preprocessing)
    with open('../pkl-files/label_encoders.pkl', 'rb') as encoders_file:
        label_encoders = pickle.load(encoders_file)
    print("✅ Label encoders loaded successfully")
    
    engine = create_engine(DATABASE_URL)
    print("Server MCP: Conectat cu succes la PostgreSQL.")
except Exception as e:
    print(f"EROARE: Serverul MCP nu s-a putut conecta la PostgreSQL sau încărca modelul: {e}")
    engine = None
    neural_network_model = None
try:
    # Load the improved Random Forest model
    with open('../pkl-files/improved_rf_model.pkl', 'rb') as model_file:
        final_rf_model = pickle.load(model_file)

    # Load the optimal threshold
    with open('../pkl-files/optimal_threshold.pkl', 'rb') as threshold_file:
        threshold_data = pickle.load(threshold_file)
        optimal_threshold = threshold_data['optimal_threshold'] if isinstance(threshold_data, dict) else threshold_data

    # Load the label encoders
    with open('../pkl-files/label_encoders.pkl', 'rb') as encoders_file:
        label_encoders = pickle.load(encoders_file)
    engine = create_engine(DATABASE_URL)
    print("Server MCP: Conectat cu succes la PostgreSQL.")
except Exception as e:
    print(f"EROARE: Serverul MCP nu s-a putut conecta la PostgreSQL: {e}")
    engine = None

# --- Modelul de Date (ce așteptăm să primim) ---
class QueryRequest(BaseModel):
    sql_query: str

def preprocess_transaction_for_nn(transaction_data):
    """
    Preprocesses a single transaction for the neural network model.
    Updated to match the neural network preprocessing pipeline.

    Args:
        transaction_data (pd.Series or dict): The transaction data.

    Returns:
        np.ndarray: The preprocessed and scaled transaction data.
    """
    # Convert to DataFrame if it's a Series or dictionary
    if isinstance(transaction_data, (pd.Series, dict)):
        df_single = pd.DataFrame([transaction_data])
    else:
        df_single = transaction_data.copy()

    # Apply the same preprocessing steps as during training
    df_single['trans_date_trans_time'] = pd.to_datetime(df_single['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
    df_single['dob'] = pd.to_datetime(df_single['dob'], format='%d/%m/%Y')
    
    # Calculate age
    df_single['age'] = df_single['trans_date_trans_time'].dt.year - df_single['dob'].dt.year
    
    # Extract time features
    df_single['transaction_hour'] = df_single['trans_date_trans_time'].dt.hour
    df_single['transaction_day'] = df_single['trans_date_trans_time'].dt.day
    df_single['transaction_month'] = df_single['trans_date_trans_time'].dt.month
    df_single['transaction_dayofweek'] = df_single['trans_date_trans_time'].dt.dayofweek
    
    # Create trigonometric features for hour (cyclical encoding)
    df_single['hour_sin'] = np.sin(2 * np.pi * df_single['transaction_hour'] / 24)
    df_single['hour_cos'] = np.cos(2 * np.pi * df_single['transaction_hour'] / 24)
    
    # Log transformation of amount (add small constant to handle zeros)
    df_single['amt_log'] = np.log1p(df_single['amt'])  # log1p = log(1+x) to handle zeros

    # Label Encode categorical columns using the saved encoders
    categorical_columns = ['category', 'gender']
    for col in categorical_columns:
        if col in df_single.columns and col in label_encoders:
            # Check if the category exists in the fitted encoder classes
            if df_single[col].iloc[0] in label_encoders[col].classes_:
                df_single[col] = label_encoders[col].transform(df_single[col])
            else:
                # Handle unseen categories - assign the first class (most common)
                df_single[col] = 0  # Default to first encoded value

    # Drop columns as specified in the neural network preprocessing
    # Based on the notebook: drop these columns for neural network
    columns_to_drop = [
        'trans_date_trans_time', 'cc_num', 'merchant', 'first', 'last', 
        'street', 'city', 'state', 'zip', 'lat', 'long', 'city_pop', 
        'job', 'trans_num', 'unix_time', 'merch_lat', 'merch_long', 'dob'
    ]
    
    # Ensure only existing columns are dropped
    columns_to_drop = [col for col in columns_to_drop if col in df_single.columns]
    df_single = df_single.drop(columns=columns_to_drop)
    
    # Drop target column if it exists
    if 'is_fraud' in df_single.columns:
        df_single = df_single.drop(columns=['is_fraud'])

    # Expected columns for neural network (based on notebook preprocessing)
    expected_columns = [
        'id', 'category', 'amt', 'gender', 'transaction_hour', 'transaction_day',
        'transaction_month', 'transaction_dayofweek', 'hour_sin', 'hour_cos', 
        'amt_log', 'age'
    ]
    
    # Ensure all expected columns exist, add missing ones with default values if needed
    for col in expected_columns:
        if col not in df_single.columns:
            df_single[col] = 0  # or some appropriate default value
    
    # Reorder columns to match training order
    df_single = df_single[expected_columns]
    
    # Drop datetime columns if they still exist
    datetime_columns = df_single.select_dtypes(include=['datetime64']).columns
    if len(datetime_columns) > 0:
        df_single = df_single.drop(columns=datetime_columns)
    
    # Scale the features using the saved scaler
    scaled_features = feature_scaler.transform(df_single)
    
    return scaled_features

@app.post("/predict_fraud")
async def predict_fraud(transaction_data: dict) -> dict:
    try:
        if neural_network_model is None:
            raise HTTPException(status_code=500, detail="Neural network model not loaded")
            
        print(f"Received transaction data: {transaction_data}")
        
        # Preprocess the data for neural network
        preprocessed_data = preprocess_transaction_for_nn(transaction_data)
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        
        # Get prediction probabilities from neural network
        prediction_proba = neural_network_model.predict(preprocessed_data, verbose=0)
        fraud_probability = float(prediction_proba[0][0])  # Neural network outputs single value
        
        # Apply optimal threshold
        prediction = int(fraud_probability >= optimal_threshold)
        
        return {
            "fraud_detected": bool(prediction), 
            "fraud_probability": float(fraud_probability),
            "threshold_used": float(optimal_threshold),
            "confidence": float(fraud_probability) if prediction else float(1 - fraud_probability),
            "model_type": "neural_network"
        }
    except Exception as e:
        print(f"Error in predict_fraud: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/generate_and_predict_transaction")
async def generate_and_predict_transaction():
    """Generate a random transaction using request_maker.py and get fraud prediction"""
    try:
        if neural_network_model is None:
            raise HTTPException(status_code=500, detail="Neural network model not loaded")
        
        # Generate random transaction using function from request_maker.py
        transaction_data = generate_random_transaction()
        
        # Preprocess the data for neural network
        preprocessed_data = preprocess_transaction_for_nn(transaction_data)
        
        # Get prediction probabilities from neural network
        prediction_proba = neural_network_model.predict(preprocessed_data, verbose=0)
        fraud_probability = float(prediction_proba[0][0])
        
        # Apply optimal threshold
        prediction = int(fraud_probability >= optimal_threshold)
        
        # Return transaction with prediction
        return {
            "transaction": transaction_data,
            "fraud_detected": bool(prediction),
            "fraud_probability": float(fraud_probability),
            "threshold_used": float(optimal_threshold),
            "confidence": float(fraud_probability) if prediction else float(1 - fraud_probability),
            "risk": float(fraud_probability * 100),  # Convert to percentage
            "model_type": "neural_network"
        }
    except Exception as e:
        import traceback
        print(f"Error in generate_and_predict_transaction: {e}")
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

# --- Endpoint-ul MCP ---
@app.post("/run_sql")
async def run_sql_query(request: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Eroare server: Conexiunea la baza de date a eșuat.")

    query = request.sql_query.strip()
    print(f"Server MCP: Am primit interogarea: {query}")

    # Măsură de siguranță simplă pentru hackathon:
    if not query.lower().startswith('select'):
        print("Server MCP: EROARE - Permitem doar interogări SELECT.")
        raise HTTPException(status_code=400, detail="Doar interogările SELECT sunt permise.")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            # Convertim rândurile în dicționare
            data = [dict(row._mapping) for row in rows]
            print(f"Server MCP: Interogare executată. Se returnează {len(data)} rânduri.")
            return {"status": "success", "data": data}
    except Exception as e:
        print(f"Server MCP: EROARE la executarea interogării: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare la executarea SQL: {e}")

# --- Endpoint de testare (ca să verificăm în browser) ---
@app.get("/")
def read_root():
    return {"mesaj": "Serverul MCP rulează cu Neural Network! Folosește endpoint-ul /run_sql pentru a trimite interogări."}

# --- Pornirea Serverului ---
if __name__ == "__main__":
    print("Se pornește Serverul MCP cu Neural Network pe http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)
import uvicorn
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel
import pandas as pd
import numpy as np
import pickle

# --- Configurare Conexiune Bază de Date ---
# Setările trebuie să se potrivească cu fișierele din 1_database
DB_USER = 'user'
DB_PASSWORD = 'pass123'
DB_HOST = 'localhost' # Se conectează la containerul Docker
DB_PORT = '5432'
DB_NAME = 'fraud_detection_db'
DATABASE_URL = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"

# --- Inițializare ---
app = FastAPI()
try:
    # Load the trained Random Forest model
    with open('../final_rf_model.pkl', 'rb') as model_file:
        fina_rf_model = pickle.load(model_file)

    # Load the label encoders
    with open('../label_encoders.pkl', 'rb') as encoders_file:
        label_encoders = pickle.load(encoders_file)
    engine = create_engine(DATABASE_URL)
    print("Server MCP: Conectat cu succes la PostgreSQL.")
except Exception as e:
    print(f"EROARE: Serverul MCP nu s-a putut conecta la PostgreSQL: {e}")
    engine = None

# --- Modelul de Date (ce așteptăm să primim) ---
class QueryRequest(BaseModel):
    sql_query: str

def preprocess_transaction(transaction_data):
    """
    Preprocesses a single transaction to the format required by the trained model.
    Updated to match the new Random Forest model with engineered features.

    Args:
        transaction_data (pd.Series or dict): The transaction data.

    Returns:
        pd.DataFrame: The preprocessed transaction data as a DataFrame.
    """
    # Convert to DataFrame if it's a Series or dictionary
    if isinstance(transaction_data, (pd.Series, dict)):
        df_single = pd.DataFrame([transaction_data])
    else:
        df_single = transaction_data.copy() # Assuming it's already a DataFrame

    # Apply the same preprocessing steps as during training
    df_single['trans_date_trans_time'] = pd.to_datetime(df_single['trans_date_trans_time'], format='%d/%m/%Y %H:%M')
    df_single['dob'] = pd.to_datetime(df_single['dob'], format='%d/%m/%Y')
    
    # Calculate age
    df_single['age'] = df_single['trans_date_trans_time'].dt.year - df_single['dob'].dt.year
    
    # Extract time features
    df_single['transaction_hour'] = df_single['trans_date_trans_time'].dt.hour
    df_single['transaction_day'] = df_single['trans_date_trans_time'].dt.day
    df_single['transaction_month'] = df_single['trans_date_trans_time'].dt.month
    df_single['transaction_dayofweek'] = df_single['trans_date_trans_time'].dt.dayofweek
    
    # Create trigonometric features for hour (cyclical encoding)
    df_single['hour_sin'] = np.sin(2 * np.pi * df_single['transaction_hour'] / 24)
    df_single['hour_cos'] = np.cos(2 * np.pi * df_single['transaction_hour'] / 24)
    
    # Log transformation of amount (add small constant to handle zeros)
    df_single['amt_log'] = np.log1p(df_single['amt'])  # log1p = log(1+x) to handle zeros

    # Label Encode categorical columns using the saved encoders
    categorical_columns = ['category', 'job', 'first', 'last', 'gender', 'street']
    for col in categorical_columns:
        if col in df_single.columns and col in label_encoders:
            # Check if the category exists in the fitted encoder classes
            if df_single[col].iloc[0] in label_encoders[col].classes_:
                df_single[col] = label_encoders[col].transform(df_single[col])
            else:
                # Handle unseen categories - assign the first class (most common)
                df_single[col] = 0  # Default to first encoded value

    # Drop irrelevant columns
    columns_to_drop = ['trans_date_trans_time', 'dob', 'trans_num','merchant','state','city', 'is_fraud', 'amt']
    # Ensure only existing columns are dropped
    columns_to_drop = [col for col in columns_to_drop if col in df_single.columns]
    df_single = df_single.drop(columns=columns_to_drop)

    # Define the exact order expected by the new model
    expected_order = [
        'id', 'cc_num', 'category', 'amt_log', 'first', 'last', 'gender', 'street', 
        'zip', 'lat', 'long', 'city_pop', 'job', 'unix_time', 'merch_lat', 'merch_long', 
        'age', 'hour_sin', 'hour_cos', 'transaction_hour', 'transaction_day', 
        'transaction_month', 'transaction_dayofweek'
    ]
    
    # Ensure all expected columns exist, add missing ones with default values if needed
    for col in expected_order:
        if col not in df_single.columns:
            df_single[col] = 0  # or some appropriate default value
    
    # Reorder columns to match training order
    df_single = df_single[expected_order]

    return df_single

@app.post("/predict_fraud")
async def predict_fraud(transaction_data: dict) -> dict:
    try:
        print(f"Received transaction data: {transaction_data}")
        preprocessed_data = preprocess_transaction(transaction_data)
        print(f"Preprocessed data columns: {list(preprocessed_data.columns)}")
        print(f"Preprocessed data shape: {preprocessed_data.shape}")
        print(f"Model expects {fina_rf_model.n_features_in_} features")
        if hasattr(fina_rf_model, 'feature_names_in_'):
            print(f"Model feature names: {list(fina_rf_model.feature_names_in_)}")
        
        prediction = fina_rf_model.predict(preprocessed_data)
        return {"fraud_detected": bool(prediction[0]), "confidence": float(prediction[0])}
    except Exception as e:
        print(f"Error in predict_fraud: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# --- Endpoint-ul MCP ---
@app.post("/run_sql")
async def run_sql_query(request: QueryRequest):
    if engine is None:
        raise HTTPException(status_code=500, detail="Eroare server: Conexiunea la baza de date a eșuat.")

    query = request.sql_query.strip()
    print(f"Server MCP: Am primit interogarea: {query}")

    # Măsură de siguranță simplă pentru hackathon:
    if not query.lower().startswith('select'):
        print("Server MCP: EROARE - Permitem doar interogări SELECT.")
        raise HTTPException(status_code=400, detail="Doar interogările SELECT sunt permise.")

    try:
        with engine.connect() as conn:
            result = conn.execute(text(query))
            rows = result.fetchall()
            # Convertim rândurile în dicționare
            data = [dict(row._mapping) for row in rows]
            print(f"Server MCP: Interogare executată. Se returnează {len(data)} rânduri.")
            return {"status": "success", "data": data}
    except Exception as e:
        print(f"Server MCP: EROARE la executarea interogării: {e}")
        raise HTTPException(status_code=500, detail=f"Eroare la executarea SQL: {e}")

# --- Endpoint de testare (ca să verificăm în browser) ---
@app.get("/")
def read_root():
    return {"mesaj": "Serverul MCP rulează! Folosește endpoint-ul /run_sql pentru a trimite interogări."}

# --- Pornirea Serverului ---
if __name__ == "__main__":
    print("Se pornește Serverul MCP pe http://127.0.0.1:8000")
    uvicorn.run(app, host="127.0.0.1", port=8000)