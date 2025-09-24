from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain.storage import LocalFileStore
import os
import time
import mimetypes
import json
import base64
import re


class FileStorage:
    """
    Class for managing file storage using LocalFileStore.

    Obiettivo: accettare QUALSIASI nome file (accenti, emoji, simboli) e
    rispettare i vincoli di LocalFileStore sui "key".

    Strategia:
      - L'API espone e accetta SEMPRE la chiave logica (Unicode) con separatore '/'.
      - Sul disco (key fisici per LocalFileStore) codifichiamo OGNI segmento
        usando URL-safe base64 senza '=' e con prefisso 'u_'.
      - Per i file preserviamo l'estensione: codifichiamo solo il "nome".
    """

    # caratteri consentiti per un key "sano" (se un segmento li contiene tutti, evitiamo l'encoding)
    _SAFE_SEGMENT_RE = re.compile(r"^[A-Za-z0-9._-]+$")

    def __init__(self, root_path: Path):
        """
        Initialize the FileStorage with the root path.

        Args:
            root_path (Path): The root path for storing files.
        """
        self.store = LocalFileStore(root_path)
        self.root_path = Path(root_path)
        self.root_path.mkdir(parents=True, exist_ok=True)

        self.metadata_store_path = self.root_path / "metadata.json"
        if not self.metadata_store_path.exists():
            with open(self.metadata_store_path, 'w', encoding='utf-8') as f:
                json.dump({}, f, ensure_ascii=False)

    # ---------------------------------------------------------------------
    # Helpers generali
    # ---------------------------------------------------------------------

    @staticmethod
    def _to_posix(key: str) -> str:
        """Normalizza i separatori in stile POSIX (/) senza alterare i caratteri."""
        return key.replace("\\", "/")

    @staticmethod
    def _b64u_encode(data: bytes) -> str:
        """base64 URL-safe senza '='."""
        return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")

    @staticmethod
    def _b64u_decode(s: str) -> bytes:
        """decode base64 URL-safe senza '=', ripristinando il padding."""
        pad = '=' * (-len(s) % 4)
        return base64.urlsafe_b64decode(s + pad)

    def _needs_encoding(self, seg: str) -> bool:
        """
        Un segmento necessita encoding se:
          - contiene char fuori dal set sicuro
          - oppure inizia già con 'u_' (per evitare collisioni col nostro marcatore)
        """
        return (not self._SAFE_SEGMENT_RE.match(seg)) or seg.startswith("u_")

    def _encode_segment(self, seg: str) -> str:
        """
        Codifica un segmento in forma sicura per LocalFileStore.
        Per i file con estensione, preserva l'estensione (ultimo '.').
        """
        # Se non serve encoding, torna il segmento così com'è.
        if not self._needs_encoding(seg):
            return seg

        # Preserva SOLTANTO l'ultima estensione (".pdf", ".zip" ecc.)
        root, ext = os.path.splitext(seg)
        # Se era solo estensione (caso patologico), codifica tutto
        if not root and ext:
            encoded = self._b64u_encode(seg.encode("utf-8"))
            return f"u_{encoded}"

        encoded_root = self._b64u_encode(root.encode("utf-8"))
        return f"u_{encoded_root}{ext}"

    def _decode_segment(self, seg: str) -> str:
        """
        Decodifica un segmento se è nel formato 'u_<base64url>[.<ext>]'.
        Altrimenti lo restituisce invariato.
        """
        if not seg.startswith("u_"):
            return seg

        # se c'è estensione, separala e decodifica solo la parte "nome"
        root, ext = os.path.splitext(seg)
        # root inizia con "u_"
        try:
            raw_b64 = root[2:]  # togli "u_"
            decoded = self._b64u_decode(raw_b64).decode("utf-8")
            return decoded + ext
        except Exception:
            # se qualcosa va storto, restituisci il segmento originale
            return seg

    def _encode_key(self, logical_key: str) -> str:
        """
        Converte la chiave logica (Unicode) in una chiave fisica sicura per LocalFileStore
        codificando OGNI segmento. Mantiene '/' come separatore.
        """
        logical_key = self._to_posix(logical_key).strip("/")
        if not logical_key:
            return ""

        parts = logical_key.split("/")
        enc_parts = [self._encode_segment(p) for p in parts]
        return "/".join(enc_parts)

    def _decode_key(self, encoded_key: str) -> str:
        """
        Inverso di _encode_key: decodifica OGNI segmento con prefisso 'u_'.
        """
        encoded_key = self._to_posix(encoded_key).strip("/")
        if not encoded_key:
            return ""
        parts = encoded_key.split("/")
        dec_parts = [self._decode_segment(p) for p in parts]
        return "/".join(dec_parts)

    # ---------------------------------------------------------------------
    # Metadata store I/O
    # ---------------------------------------------------------------------

    def _load_metadata_store(self) -> Dict[str, Any]:
        with open(self.metadata_store_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _save_metadata_store(self, metadata_store: Dict[str, Any]) -> None:
        with open(self.metadata_store_path, 'w', encoding='utf-8') as f:
            json.dump(metadata_store, f, indent=4, ensure_ascii=False)

    # ---------------------------------------------------------------------
    # API principali
    # ---------------------------------------------------------------------

    def save_file(self, key: str, content: bytes, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Salva un file usando la chiave logica (qualsiasi Unicode).
        Internamente usiamo chiavi fisiche "safe" per LocalFileStore.
        """
        logical_key = self._to_posix(key)
        encoded_key = self._encode_key(logical_key)

        # Scrittura su filesystem via LocalFileStore (niente '%' → ok)
        self.store.mset([(encoded_key, content)])

        # Metadati
        file_path = Path(self.store.root_path) / encoded_key
        mime_type, _ = mimetypes.guess_type(str(file_path))

        rec = {
            "name": logical_key,              # chiave logica (leggibile)
            "encoded_name": encoded_key,      # percorso fisico
            "size": file_path.stat().st_size,
            "modified_time": time.ctime(file_path.stat().st_mtime),
            "created_time": time.ctime(file_path.stat().st_ctime),
            "path": str(file_path),
            "mime_type": mime_type,
            "custom_metadata": custom_metadata or {},
        }

        store = self._load_metadata_store()
        # Per i file, la chiave nel metadata store è l'encoded (allineata al disco)
        store[encoded_key] = rec
        self._save_metadata_store(store)

    def get_file(self, key: str) -> Optional[bytes]:
        """
        Retrieve a file from the storage (passa SEMPRE la chiave logica).
        """
        encoded_key = self._encode_key(self._to_posix(key))
        return self.store.mget([encoded_key])[0]

    def delete_file(self, key: str) -> None:
        """
        Delete a file from the storage and remove its metadata (chiave logica in input).
        """
        encoded_key = self._encode_key(self._to_posix(key))
        self.store.mdelete([encoded_key])

        store = self._load_metadata_store()
        if encoded_key in store:
            del store[encoded_key]
        else:
            # retrocompatibilità: se in passato fossero state salvate chiavi logiche
            store.pop(self._to_posix(key), None)
        self._save_metadata_store(store)

    def list_files(self, directory: Optional[str] = None) -> List[str]:
        """
        List all files in a specific directory or globally if no directory is specified.
        Ritorna SEMPRE chiavi LOGICHE (decodificate).
        """
        if directory:
            logical_dir = self._to_posix(directory).strip("/")
            enc_dir = self._encode_key(logical_dir)
            prefix = enc_dir + "/"
            keys = [k for k in self.store.yield_keys() if k.startswith(prefix)]
        else:
            keys = list(self.store.yield_keys())

        return [self._decode_key(k) for k in keys]

    def update_file(self, key: str, content: bytes, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing file in the storage along with its metadata (chiave logica).
        """
        self.save_file(key, content, custom_metadata)

    def save_file_metadata(self, file_id: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Save metadata for a specific file (file_id = chiave logica).
        """
        encoded_key = self._encode_key(self._to_posix(file_id))
        store = self._load_metadata_store()
        if encoded_key not in store:
            # compat: prova con chiave logica pura
            if file_id not in store:
                raise KeyError(f"File '{file_id}' does not exist in metadata store.")
            target = file_id
        else:
            target = encoded_key

        store[target]['custom_metadata'] = custom_metadata or {}
        self._save_metadata_store(store)

    def update_file_metadata(self, file_id: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Aggiorna (merge) i metadati personalizzati di un file.
        Se il record non esiste in metadata.json, viene creato on-the-fly.
        """
        encoded_key = self._encode_key(self._to_posix(file_id))
        store = self._load_metadata_store()

        # Se manca il record, prova a costruirlo dai metadati filesystem
        if encoded_key not in store:
            base = self.get_file_metadata(file_id)
            if not base:
                raise KeyError(
                    f"File '{file_id}' non trovato sul filesystem; impossibile creare metadati."
                )
            store[encoded_key] = base

        existing_custom = store[encoded_key].get("custom_metadata", {})
        if not isinstance(existing_custom, dict):
            existing_custom = {}

        merged = {**existing_custom, **(custom_metadata or {})}
        store[encoded_key]["custom_metadata"] = merged
        self._save_metadata_store(store)

    def get_file_metadata(self, key: str) -> dict:
        """
        Retrieve the metadata of a specific file (chiave logica).
        """
        logical_key = self._to_posix(key)
        encoded_key = self._encode_key(logical_key)
        file_path = Path(self.store.root_path) / encoded_key
        if not file_path.exists():
            return {}
        mime_type, _ = mimetypes.guess_type(str(file_path))
        store = self._load_metadata_store()

        rec = store.get(encoded_key, {})
        # se mancano, ricostruisci base record
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
        else:
            # allinea i campi dinamici
            rec["name"] = logical_key
            rec["encoded_name"] = encoded_key
            rec["path"] = str(file_path)
            rec["mime_type"] = mime_type

        return rec

    # ---------------------------------------------------------------------
    # Metadati directory (le directory restano chiavi LOGICHE nel metadata)
    # ---------------------------------------------------------------------

    def save_directory_metadata(self, directory: str, custom_metadata: Dict[str, Any]) -> None:
        store = self._load_metadata_store()
        logical_dir = self._to_posix(directory).strip("/")
        store[logical_dir] = custom_metadata or {}
        self._save_metadata_store(store)

    def update_directory_metadata(self, directory: str, custom_metadata: Dict[str, Any]) -> None:
        store = self._load_metadata_store()
        logical_dir = self._to_posix(directory).strip("/")
        existing = store.get(logical_dir, {})
        if not isinstance(existing, dict):
            existing = {}
        store[logical_dir] = {**existing, **(custom_metadata or {})}
        self._save_metadata_store(store)

    def get_directory_metadata(self, directory: str) -> dict:
        store = self._load_metadata_store()
        logical_dir = self._to_posix(directory).strip("/")
        return store.get(logical_dir, {})

    def get_directory_metadata_bulk(self, directories: List[str]) -> Dict[str, dict]:
        store = self._load_metadata_store()
        dirs = [self._to_posix(d).strip("/") for d in directories]
        return {d: store.get(d, {}) for d in dirs}

    def list_directories(self) -> List[str]:
        """
        Elenca *solo* le directory registrate nel metadata store,
        indipendentemente dal fatto che contengano file o meno.
        (Le directory restano chiavi LOGICHE, non encodate.)
        """
        metadata_store = self._load_metadata_store()
        directories = [
            key for key, meta in metadata_store.items()
            if isinstance(meta, dict) and "size" not in meta
        ]
        return sorted(directories)

    def delete_directory_metadata(self, directory: str) -> None:
        store = self._load_metadata_store()
        logical_dir = self._to_posix(directory).strip("/")
        if logical_dir in store:
            del store[logical_dir]
            self._save_metadata_store(store)
        else:
            raise KeyError(f"Metadata for directory '{logical_dir}' does not exist.")

    # ---------------------------------------------------------------------
    # Ricerca / Filtri
    # ---------------------------------------------------------------------

    def search_files(self, query: str, directory: Optional[str] = None) -> List[str]:
        """
        Search for files matching the query within a specific directory or globally.
        Ritorna chiavi LOGICHE.
        """
        store = self._load_metadata_store()
        results = []
        dir_prefix = None
        if directory:
            dir_prefix = self._to_posix(directory).strip("/") + "/"

        for enc_key, meta in store.items():
            # file = record con 'size'
            if not (isinstance(meta, dict) and "size" in meta):
                continue

            logical_key = meta.get("name")
            if not logical_key:
                # fallback se record vecchio
                logical_key = self._decode_key(enc_key)

            if dir_prefix and not logical_key.startswith(dir_prefix):
                continue

            hay = json.dumps(meta, ensure_ascii=False).lower()
            if (query.lower() in logical_key.lower()) or (query.lower() in hay):
                results.append(logical_key)

        return results

    def filter_files(self, mime_type: Optional[str] = None,
                     min_size: Optional[int] = None,
                     max_size: Optional[int] = None) -> List[str]:
        """
        Filter files based on MIME type and size.
        Ritorna chiavi LOGICHE.
        """
        store = self._load_metadata_store()
        results: List[str] = []

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
            results.append(logical_key)

        return results
