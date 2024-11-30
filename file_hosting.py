from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
import os

from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Permetti tutte le origini
    allow_credentials=True,
    allow_methods=["*"],  # Permetti tutti i metodi (GET, POST, OPTIONS, ecc.)
    allow_headers=["*"],  # Permetti tutti gli headers
)
# Specifica la directory che vuoi servire
DIRECTORY_TO_SERVE = "data"

# Verifica che la directory esista
if not os.path.isdir(DIRECTORY_TO_SERVE):
    raise Exception(f"La directory {DIRECTORY_TO_SERVE} non esiste")

@app.get("/files/{file_path:path}")
async def serve_file(file_path: str):
    # Costruisci il percorso completo del file
    full_path = os.path.join(DIRECTORY_TO_SERVE, file_path)
    # Normalizza il percorso per evitare path traversal
    full_path = os.path.normpath(full_path)
    # Verifica che il percorso sia all'interno della directory servita
    if not full_path.startswith(os.path.abspath(DIRECTORY_TO_SERVE)):
        raise HTTPException(status_code=403, detail="Accesso non autorizzato")
    # Verifica che il file esista e sia un file
    if os.path.isfile(full_path):
        return FileResponse(full_path)
    else:
        raise HTTPException(status_code=404, detail="File non trovato")

@app.get("/list/{dir_path:path}")
async def list_directory(dir_path: str = ""):
    # Costruisci il percorso completo della directory
    full_path = os.path.join(DIRECTORY_TO_SERVE, dir_path)
    # Normalizza il percorso per evitare path traversal
    full_path = os.path.normpath(full_path)
    # Verifica che il percorso sia all'interno della directory servita
    if not full_path.startswith(os.path.abspath(DIRECTORY_TO_SERVE)):
        raise HTTPException(status_code=403, detail="Accesso non autorizzato")
    # Verifica che la directory esista
    if os.path.isdir(full_path):
        items = os.listdir(full_path)
        return {"files": items}
    else:
        raise HTTPException(status_code=404, detail="Directory non trovata")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8093)
