import uvicorn
from fastapi import FastAPI, HTTPException
from sqlalchemy import create_engine, text
from pydantic import BaseModel

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
    engine = create_engine(DATABASE_URL)
    print("Server MCP: Conectat cu succes la PostgreSQL.")
except Exception as e:
    print(f"EROARE: Serverul MCP nu s-a putut conecta la PostgreSQL: {e}")
    engine = None

# --- Modelul de Date (ce așteptăm să primim) ---
class QueryRequest(BaseModel):
    sql_query: str

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