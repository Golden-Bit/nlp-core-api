#!/usr/bin/env python3
"""
Demo rapido delle API di data_stores.

Assicurati che il server sia in esecuzione su http://127.0.0.1:8100
e che il file example.txt esista nella stessa cartella di questo script
(o verrà creato al volo).
"""

import json
import os
import requests
from pprint import pprint

BASE_URL = "http://127.0.0.1:8100/data_stores"

# ----------------------------------------------------------------------
# 1. Crea directory con metadati
# ----------------------------------------------------------------------
dir_path = "docs/demo"
print(f"\n1) Creo directory '{dir_path}' ...")
resp = requests.post(
    f"{BASE_URL}/create_directory",
    data={
        "directory": dir_path,
        "description": "Directory di esempio creata via API",
        "extra_metadata": json.dumps({"project": "Sample", "owner": "Alice"}),
    },
)
resp.raise_for_status()
pprint(resp.json())  # DirectoryMetadata

# ----------------------------------------------------------------------
# 2. Leggi metadati directory
# ----------------------------------------------------------------------
print("\n2) Recupero i metadati della directory ...")
resp = requests.get(f"{BASE_URL}/directories")
resp.raise_for_status()
# Filtra solo la nostra
dir_meta = next(d for d in resp.json() if d["path"] == dir_path)
pprint(dir_meta)

# ----------------------------------------------------------------------
# 3. Aggiorna (merge) metadati directory
# ----------------------------------------------------------------------
print("\n3) Aggiorno (merge) i metadati della directory ...")
resp = requests.put(
    f"{BASE_URL}/directory/metadata/{dir_path}",
    data={
        "description": "Descrizione aggiornata",
        "extra_metadata": json.dumps({"status": "in-progress"}),
    },
)
resp.raise_for_status()
pprint(resp.json())

# ----------------------------------------------------------------------
# 4. Carica un file nella directory
# ----------------------------------------------------------------------
print("\n4) Carico il file 'example.txt' ...")
# Se non esiste, creiamolo al volo
if not os.path.exists("example.txt"):
    with open("example.txt", "w", encoding="utf-8") as fh:
        fh.write("Questo è un file di test caricato via API.\n")

with open("example.txt", "rb") as fh:
    resp = requests.post(
        f"{BASE_URL}/upload",
        files={"file": fh},
        data={
            "subdir": dir_path,
            "file_description": "File di test",
            "extra_metadata": json.dumps({"version": 1}),
        },
    )
resp.raise_for_status()
file_meta = resp.json()
pprint(file_meta)

file_id = file_meta["name"]  # es. "docs/demo/example.txt"

# ----------------------------------------------------------------------
# 5. Recupera metadati del file
# ----------------------------------------------------------------------
print("\n5) Recupero i metadati del file ...")
resp = requests.get(f"{BASE_URL}/metadata/{file_id}")
resp.raise_for_status()
pprint(resp.json())

# ----------------------------------------------------------------------
# 6. Aggiorna (merge) metadati file
# ----------------------------------------------------------------------
print("\n6) Aggiorno (merge) i metadati del file ...")
resp = requests.put(
    f"{BASE_URL}/file/metadata/{file_id}",
    data={
        "file_description": "File di test – versione aggiornata",
        "extra_metadata": json.dumps({"version": 2, "reviewed": True}),
    },
)
resp.raise_for_status()
pprint(resp.json())

print("\n✓ Demo completata senza errori")
