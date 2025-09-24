from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain.storage import LocalFileStore
import os
import time
import mimetypes
import json
from urllib.parse import quote, unquote


class FileStorage:
    """
    Gestione file con LocalFileStore, supportando nomi con QUALSIASI carattere.
    - I "key" visibili all'esterno restano gli originali (Unicode).
    - Sui path fisici su disco usiamo percent-encoding per ogni segmento
      (così evitiamo caratteri vietati su Windows/macOS e problemi con separatori).
    """

    def __init__(self, root_path: Path):
        self.store = LocalFileStore(root_path)
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

        self.metadata_store_path = self.root_path / "metadata.json"
        if not self.metadata_store_path.exists():
            with open(self.metadata_store_path, "w", encoding="utf-8") as f:
                json.dump({}, f, ensure_ascii=False)

    # ------------------------- Encoding helpers -------------------------

    @staticmethod
    def _to_posix(key: str) -> str:
        """Normalizza i separatori in stile POSIX (/) senza toccare i caratteri."""
        return key.replace("\\", "/")

    @staticmethod
    def _encode_segment(seg: str) -> str:
        """
        Percent-encode di un singolo segmento di path.
        Manteniamo solo [A-Za-z0-9-._()] come 'safe' (spazi e tutto il resto → %XX).
        """
        return quote(seg, safe="-._()")

    @staticmethod
    def _decode_segment(seg: str) -> str:
        return unquote(seg)

    def _encode_key(self, logical_key: str) -> str:
        """
        Converte la chiave "logica" (visibile all'esterno) in path fisico sicuro.
        Preserva la struttura a directory: codifica ogni segmento separatamente.
        Esempio: "ctx/fi le/bozza(1).pdf" → "ctx/fi%20le/bozza(1).pdf"
        """
        logical_key = self._to_posix(logical_key)
        parts = [self._encode_segment(p) for p in logical_key.split("/")]
        return "/".join(parts)

    def _decode_key(self, encoded_key: str) -> str:
        """Operazione inversa di _encode_key."""
        parts = [self._decode_segment(p) for p in encoded_key.split("/")]
        return "/".join(parts)

    # ------------------------- Metadata I/O -------------------------

    def _load_metadata_store(self) -> Dict[str, Any]:
        with open(self.metadata_store_path, "r", encoding="utf-8") as f:
            return json.load(f)

    def _save_metadata_store(self, metadata_store: Dict[str, Any]) -> None:
        with open(self.metadata_store_path, "w", encoding="utf-8") as f:
            json.dump(metadata_store, f, indent=4, ensure_ascii=False)

    # ------------------------- API principali -------------------------

    def save_file(self, key: str, content: bytes, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Salva un file usando la chiave *logica* (originale). Internamente codifichiamo il path.
        """
        logical_key = self._to_posix(key)
        encoded_key = self._encode_key(logical_key)

        # Scrittura sullo store fisico
        self.store.mset([(encoded_key, content)])

        # Metadati
        file_path = Path(self.store.root_path) / encoded_key
        mime_type, _ = mimetypes.guess_type(str(file_path))  # ok anche con %XX

        rec = {
            "name": logical_key,                 # chiave logica (originale, Unicode)
            "encoded_name": encoded_key,         # path fisico (percent-encoded)
            "size": file_path.stat().st_size,
            "modified_time": time.ctime(file_path.stat().st_mtime),
            "created_time": time.ctime(file_path.stat().st_ctime),
            "path": str(file_path),
            "mime_type": mime_type,
            "custom_metadata": custom_metadata or {},
        }

        store = self._load_metadata_store()
        # Chiave nel metadata store: usiamo **encoded_key** per allinearci al disco
        store[encoded_key] = rec
        self._save_metadata_store(store)

    def get_file(self, key: str) -> Optional[bytes]:
        """Ritorna i bytes del file dato il path logico."""
        encoded_key = self._encode_key(self._to_posix(key))
        return self.store.mget([encoded_key])[0]

    def delete_file(self, key: str) -> None:
        """Elimina file e metadati. Accetta path logico."""
        encoded_key = self._encode_key(self._to_posix(key))
        self.store.mdelete([encoded_key])

        store = self._load_metadata_store()
        if encoded_key in store:
            del store[encoded_key]
        else:
            # Fallback compatibilità: vecchie installazioni potrebbero aver usato chiavi "logiche"
            store.pop(self._to_posix(key), None)
        self._save_metadata_store(store)

    def list_files(self, directory: Optional[str] = None) -> List[str]:
        """
        Elenca file (ritorna CHIAVI LOGICHE).
        Se `directory` è passata, filtriamo sul prefisso *encoded* corrispondente.
        """
        if directory:
            logical_dir = self._to_posix(directory).rstrip("/") + "/"
            encoded_dir = self._encode_key(logical_dir)
            keys = [k for k in self.store.yield_keys() if k.startswith(encoded_dir)]
        else:
            keys = list(self.store.yield_keys())

        # Decodifica ogni chiave prima di restituirla
        return [self._decode_key(k) for k in keys]

    def update_file(self, key: str, content: bytes, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        """Aggiorna contenuto (e opzionalmente metadati) di un file logico."""
        self.save_file(key, content, custom_metadata)

    # ------------------------- Metadati file -------------------------

    def save_file_metadata(self, file_id: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Sovrascrive i metadati custom del file indicato (per chiave logica).
        """
        encoded_key = self._encode_key(self._to_posix(file_id))
        store = self._load_metadata_store()

        if encoded_key not in store:
            # Compat vecchie chiavi
            if file_id in store:
                encoded_key = file_id
            else:
                raise KeyError(f"File '{file_id}' non esiste nel metadata store.")

        store[encoded_key]["custom_metadata"] = custom_metadata or {}
        self._save_metadata_store(store)

    def update_file_metadata(self, file_id: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Merge dei metadati custom del file (chiave logica).
        """
        encoded_key = self._encode_key(self._to_posix(file_id))
        store = self._load_metadata_store()

        # Se manca la voce, prova a generarla dai metadati di base
        if encoded_key not in store:
            # Compat vecchie chiavi logiche
            fallback_key = file_id if file_id in store else None
            base = self.get_file_metadata(file_id)
            if not base:
                raise KeyError(f"File '{file_id}' non trovato.")
            store[fallback_key or encoded_key] = base

        target_key = encoded_key if encoded_key in store else file_id
        existing_custom = store[target_key].get("custom_metadata", {})
        if not isinstance(existing_custom, dict):
            existing_custom = {}
        store[target_key]["custom_metadata"] = {**existing_custom, **(custom_metadata or {})}
        self._save_metadata_store(store)

    def get_file_metadata(self, key: str) -> dict:
        """
        Ritorna il record metadati del file (chiave logica).
        """
        logical_key = self._to_posix(key)
        encoded_key = self._encode_key(logical_key)

        file_path = Path(self.store.root_path) / encoded_key
        if not file_path.exists():
            return {}

        mime_type, _ = mimetypes.guess_type(str(file_path))
        store = self._load_metadata_store()
        rec = store.get(encoded_key)

        # Se non presente (vecchi record), crea on-the-fly
        if not rec:
            rec = {
                "name": logical_key,
                "encoded_name": encoded_key,
                "size": file_path.stat().st_size,
                "modified_time": time.ctime(file_path.stat().st_mtime),
                "created_time": time.ctime(file_path.stat().st_ctime),
                "path": str(file_path),
                "mime_type": mime_type,
                "custom_metadata": {},
            }

        # Garantisci che 'name' sia sempre il percorso logico (decodificato)
        rec["name"] = logical_key
        rec["encoded_name"] = encoded_key
        rec["path"] = str(file_path)
        rec["mime_type"] = mime_type
        return rec

    # ------------------------- Metadati directory -------------------------

    def save_directory_metadata(self, directory: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Salva metadati per una directory (chiave = path logico della directory).
        """
        logical_dir = self._to_posix(directory)
        store = self._load_metadata_store()
        store[logical_dir] = custom_metadata or {}
        self._save_metadata_store(store)

    def update_directory_metadata(self, directory: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Merge dei metadati di directory (chiave = path logico).
        """
        logical_dir = self._to_posix(directory)
        store = self._load_metadata_store()
        existing = store.get(logical_dir, {})
        if not isinstance(existing, dict):
            existing = {}
        store[logical_dir] = {**existing, **(custom_metadata or {})}
        self._save_metadata_store(store)

    def get_directory_metadata(self, directory: str) -> dict:
        logical_dir = self._to_posix(directory)
        store = self._load_metadata_store()
        return store.get(logical_dir, {})

    def get_directory_metadata_bulk(self, directories: List[str]) -> Dict[str, dict]:
        store = self._load_metadata_store()
        dirs = [self._to_posix(d) for d in directories]
        return {d: store.get(d, {}) for d in dirs}

    def list_directories(self) -> List[str]:
        """
        Ritorna le *sole* directory registrate nel metadata store (chiavi senza 'size').
        Le directory restano chiavi logiche (non encodate).
        """
        store = self._load_metadata_store()
        directories = [
            key for key, meta in store.items()
            if isinstance(meta, dict) and "size" not in meta
        ]
        return sorted(directories)

    def delete_directory_metadata(self, directory: str) -> None:
        logical_dir = self._to_posix(directory)
        store = self._load_metadata_store()
        if logical_dir in store:
            del store[logical_dir]
            self._save_metadata_store(store)
        else:
            raise KeyError(f"Metadata for directory '{logical_dir}' does not exist.")

    # ------------------------- Ricerca / Filtri -------------------------

    def search_files(self, query: str, directory: Optional[str] = None) -> List[str]:
        """
        Cerca per sottostringa su chiave logica e metadati. Ritorna CHIAVI LOGICHE.
        """
        store = self._load_metadata_store()
        results = []
        dir_prefix = self._to_posix(directory).rstrip("/") + "/" if directory else None

        for enc_key, meta in store.items():
            # Filtra solo record file (quelli con 'size')
            if not (isinstance(meta, dict) and "size" in meta):
                continue

            logical_key = meta.get("name") or self._decode_key(enc_key)
            if dir_prefix and not logical_key.startswith(dir_prefix):
                continue

            hay = json.dumps(meta, ensure_ascii=False).lower()
            if (query.lower() in logical_key.lower()) or (query.lower() in hay):
                results.append(logical_key)

        return results

    def filter_files(
        self,
        mime_type: Optional[str] = None,
        min_size: Optional[int] = None,
        max_size: Optional[int] = None,
    ) -> List[str]:
        """
        Filtra per mimetype e dimensione. Ritorna CHIAVI LOGICHE.
        """
        store = self._load_metadata_store()
        out: List[str] = []

        for enc_key, meta in store.items():
            if not (isinstance(meta, dict) and "size" in meta):
                continue

            if mime_type and meta.get("mime_type") != mime_type:
                continue

            size = meta.get("size")
            if size is not None:
                if min_size is not None and size < min_size:
                    continue
                if max_size is not None and size > max_size:
                    continue

            logical_key = meta.get("name") or self._decode_key(enc_key)
            out.append(logical_key)

        return out
