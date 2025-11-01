#!/bin/bash
# Noul run.sh: Pornește PostgreSQL, Query Server (pe 8001) și Simulatorul Live

# --- Configurare ---
# Oprește scriptul imediat dacă o comandă eșuează
set -e

echo "============================================="
echo "=== START SCRIPT: PORNIRE TOTALĂ SISTEM ==="
echo "============================================="

# --- Găsește directorul principal ---
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# --- 1. Verificarea și Pornirea Serviciului Docker ---
echo "INFO: Se verifică statusul serviciului Docker..."
sudo systemctl unmask docker.service > /dev/null 2>&1 || true
sudo systemctl start docker
sleep 2
if ! sudo docker ps > /dev/null; then
    echo "EROARE: Serviciul Docker (daemon) nu rulează."
    exit 1
fi
echo "INFO: Serviciul Docker rulează."

# --- 2. Navigarea la Directorul Bazei de Date (Contine docker-compose.yml) ---
DB_DIR="$SCRIPT_DIR/1_database"
cd "$DB_DIR"
echo "INFO: Se lucrează în directorul: $(pwd)"

# --- 3. Oprirea și Ștergerea Containerelor Vechi ---
echo "INFO: Se opresc și se șterg containerele vechi (DB + Query Server)..."
sudo docker compose down -v

# --- 4. Pornirea Containerelor (DB & MCP Server) ---
echo "INFO: Se pornește PostgreSQL (DB) și Query Server (MCP) [accesibil la portul 8001]..."
# Această comandă pornește AMBELE servicii (DB și query-server)
sudo docker compose up -d

echo "INFO: Se așteaptă ca baza de date să pornească (15 secunde)..."
sleep 15

# --- 5. Încărcarea Datelor (FULL LOAD) ---
echo "INFO: Se rulează scriptul 'load_data.py' pentru a încărca CSV-ul (FULL LOAD)..."
# Argumentul --full forțează încărcarea întregului fișier
python3 load_data.py --full
echo "INFO: Pasul 1 (Baza de Date) este finalizat."

# --- 6. Pornirea Simulatorului Live (Rulând pe Host) ---
SIM_DIR="$SCRIPT_DIR/5_live_simulator"
if [ ! -f "$SIM_DIR/simulator.py" ]; then
    echo "WARN: Nu am găsit '5_live_simulator/simulator.py'. Se sare peste."
else
    cd "$SIM_DIR"
    echo "INFO: Se pornește Simulatorul Live (Pasul 5) în fundal..."
    # Pornim cu nohup pentru a rula în fundal
    nohup python3 simulator.py > simulator.log 2>&1 & echo $! > simulator.pid
    sleep 1
    echo "INFO: Simulatorul rulează în fundal (PID: $(cat simulator.pid))."
    echo "      (Vezi log-ul în '5_live_simulator/simulator.log')"
fi

# --- 7. Mesaj Final ---
cd "$SCRIPT_DIR"
echo "============================================="
echo "=== SUCCES: TOATE SERVICIILE AU PORNIT! ==="
echo "============================================="
echo "Serverul MCP (Query Server) este accesibil la: http://127.0.0.1:8001/run_sql"
echo "Pentru a opri totul, rulează: ./stop.sh"