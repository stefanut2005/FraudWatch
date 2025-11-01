#!/bin/bash

# --- Configurare ---
# Oprește scriptul imediat dacă o comandă eșuează
set -e

echo "============================================="
echo "=== START SCRIPT: PASUL 1 (Baza de Date) ==="
echo "============================================="

# --- 1. Verificarea și Pornirea Serviciului Docker ---
echo "INFO: Se verifică statusul serviciului Docker..."

# Încearcă să pornească serviciul. 
# Mai întâi, îl demascăm, pentru a rezolva eroarea 'masked'.
sudo systemctl unmask docker.service > /dev/null 2>&1 || true
sudo systemctl start docker

# Așteaptă 2 secunde și verifică dacă rulează
sleep 2
if ! sudo docker ps > /dev/null; then
    echo "EROARE: Serviciul Docker (daemon) nu rulează."
    echo "Te rog instalează Docker și rulează: sudo systemctl start docker"
    exit 1
fi
echo "INFO: Serviciul Docker rulează."


# --- 2. Navigarea la Directorul Bazei de Date ---
# Găsește directorul în care se află acest script
SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
DB_DIR="$SCRIPT_DIR/1_database"

if [ ! -d "$DB_DIR" ]; then
    echo "EROARE: Nu am găsit directorul '1_database'."
    exit 1
fi

cd "$DB_DIR"
echo "INFO: Se lucrează în directorul: $(pwd)"


# --- 3. Oprirea și Ștergerea Containerelor Vechi ---
echo "INFO: Se opresc și se șterg containerele vechi (dacă există)..."
# Folosim 'docker compose' (care citește docker-compose.yml)
sudo docker compose down -v


# --- 4. Pornirea Containerului Bazei de Date ---
echo "INFO: Se pornește noul container PostgreSQL..."
sudo docker compose up -d

echo "INFO: Se așteaptă 10 secunde ca baza de date să pornească..."
sleep 10


# --- 5. Încărcarea Datelor ---
echo "INFO: Se rulează scriptul 'load_data.py' pentru a încărca CSV-ul..."
python3 load_data.py


echo "============================================="
echo "=== SUCCES: PASUL 1 ESTE FINALIZAT! ==="
echo "============================================="