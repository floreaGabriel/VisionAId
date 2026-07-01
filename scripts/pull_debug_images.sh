#!/bin/bash

# Calea catre ADB (detectata automat)
ADB_PATH="$HOME/Library/Android/sdk/platform-tools/adb"

# Numele pachetului aplicatiei
PACKAGE_NAME="com.florea_gabriel.impairedhelpapp"

# Calea de pe telefon unde sunt salvate imaginile de debug
DEVICE_PATH="/sdcard/Android/data/$PACKAGE_NAME/files/Pictures/Debug_Registration/"

# Calea locala (in proiect) unde le copiem
LOCAL_PATH="./debug_registration_images/"

echo "------------------------------------------------"
echo "🔍 ADB Status:"
"$ADB_PATH" devices
echo "------------------------------------------------"
echo "📦 Tragere imagini de debug..."
echo "De pe telefon: $DEVICE_PATH"
echo "In folderul: $LOCAL_PATH"
echo "------------------------------------------------"

# Creaza folderul local daca nu exista
mkdir -p "$LOCAL_PATH"

# Copiaza fisierele
"$ADB_PATH" pull "$DEVICE_PATH" "$LOCAL_PATH"

if [ $? -eq 0 ]; then
    echo "✅ Gata! Imaginile sunt in folderul 'debug_registration_images'"
    echo "Verifica acum folderul pentru a vedea ce 'vede' aplicatia."
    
    # Deschide folderul automat in Finder
    open "$LOCAL_PATH"
else
    echo "❌ Ceva nu a mers. Verifica daca:"
    echo "1. Ai rulat inregistrarea unui obiect nou in aplicatie?"
    echo "2. Telefonul este deblocat?"
fi
echo "------------------------------------------------"
