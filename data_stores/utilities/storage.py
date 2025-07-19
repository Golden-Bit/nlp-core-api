from pathlib import Path
from typing import List, Optional, Dict, Any
from langchain.storage import LocalFileStore
import os
import time
import mimetypes
import json


class FileStorage:
    """
    Class for managing file storage using LocalFileStore.
    """

    def __init__(self, root_path: Path):
        """
        Initialize the FileStorage with the root path.

        Args:
            root_path (Path): The root path for storing files.
        """
        self.store = LocalFileStore(root_path)
        self.metadata_store_path = root_path / "metadata.json"
        if not self.metadata_store_path.exists():
            with open(self.metadata_store_path, 'w') as f:
                json.dump({}, f)

        # -- NEW --------------------------------------------------------------
    def _get_metadata_store_cached(self) -> Dict[str, Any]:
        """
        Ritorna un riferimento _in‑memory_ al metadata store.
        Viene caricato solo la prima volta per la durata della request.
        """
        if not hasattr(self, "_cached_meta"):
            self._cached_meta = self._load_metadata_store()
        return self._cached_meta

    def _load_metadata_store(self) -> Dict[str, Any]:
        """
        Load the metadata store from the JSON file.

        Returns:
            Dict[str, Any]: The metadata store as a dictionary.
        """
        with open(self.metadata_store_path, 'r') as f:
            return json.load(f)

    def _save_metadata_store(self, metadata_store: Dict[str, Any]) -> None:
        """
        Save the metadata store to the JSON file.

        Args:
            metadata_store (Dict[str, Any]): The metadata store to save.
        """
        with open(self.metadata_store_path, 'w') as f:
            json.dump(metadata_store, f, indent=4)

    def _normalize_key(self, key: str) -> str:
        """
        Normalize the file path to use forward slashes.

        Args:
            key (str): The file path to normalize.

        Returns:
            str: The normalized file path.
        """
        return key.replace("\\", "/")

    def save_file(self, key: str, content: bytes, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        key = self._normalize_key(key)

        self.store.mset([(key, content)])
        metadata_store = self._load_metadata_store()
        file_metadata = self.get_file_metadata(key)
        if custom_metadata:
            file_metadata["custom_metadata"] = custom_metadata
        metadata_store[key] = file_metadata
        self._save_metadata_store(metadata_store)

    def get_file(self, key: str) -> Optional[bytes]:
        """
        Retrieve a file from the storage.

        Args:
            key (str): The key (path) of the file to retrieve.

        Returns:
            Optional[bytes]: The content of the file or None if the file does not exist.
        """
        return self.store.mget([key])[0]

    def delete_file(self, key: str) -> None:
        """
        Delete a file from the storage and remove its metadata.

        Args:
            key (str): The key (path) of the file to delete.
        """
        self.store.mdelete([key])
        metadata_store = self._load_metadata_store()
        if key in metadata_store:
            del metadata_store[key]
        self._save_metadata_store(metadata_store)

    def list_files(self, directory: Optional[str] = None) -> List[str]:
        """
        List all files in a specific directory or globally if no directory is specified.

        Args:
            directory (Optional[str]): The directory to list files from. If None, lists all files.

        Returns:
            List[str]: A list of file keys (paths).
        """
        directory = self._normalize_key(directory) if directory else None
        if directory:
            return [key for key in self.store.yield_keys() if key.startswith(directory)]
        return list(self.store.yield_keys())

    def update_file(self, key: str, content: bytes, custom_metadata: Optional[Dict[str, Any]] = None) -> None:
        """
        Update an existing file in the storage along with its metadata.

        Args:
            key (str): The key (path) of the file to update.
            content (bytes): The new content of the file.
            custom_metadata (Optional[Dict[str, Any]]): Custom metadata provided by the user.
        """
        self.save_file(key, content, custom_metadata)

    def save_file_metadata(self, file_id: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Save metadata for a specific file.

        Args:
            file_id (str): The file path.
            custom_metadata (Dict[str, Any]): Custom metadata provided by the user.
        """
        metadata_store = self._load_metadata_store()
        if file_id not in metadata_store:
            raise KeyError(f"File '{file_id}' does not exist in metadata store.")
        metadata_store[file_id]['custom_metadata'] = custom_metadata
        self._save_metadata_store(metadata_store)

    def update_file_metadata(
        self, file_id: str, custom_metadata: Dict[str, Any]
    ) -> None:
        """
        Aggiorna (merge) i metadati personalizzati di un file.
        Se il record non esiste ancora in metadata.json, viene creato
        partendo dai metadati “di base” del file sul filesystem.
        """
        metadata_store = self._load_metadata_store()

        # Recupera record corrente (se manca, prova a generarlo da zero)
        if file_id not in metadata_store:
            base_meta = self.get_file_metadata(file_id)
            if not base_meta:
                raise KeyError(
                    f"File '{file_id}' non trovato sul filesystem; impossibile "
                    "creare metadati."
                )
            metadata_store[file_id] = base_meta

        # Inizializza struttura custom se assente o non-dict
        existing_custom = metadata_store[file_id].get("custom_metadata", {})
        if not isinstance(existing_custom, dict):
            existing_custom = {}

        # Merge: i valori nuovi sovrascrivono quelli vecchi
        merged = {**existing_custom, **custom_metadata}
        metadata_store[file_id]["custom_metadata"] = merged
        self._save_metadata_store(metadata_store)

    def get_file_metadata(self, key: str) -> dict:
        """
        Retrieve the metadata of a specific file.

        Args:
            key (str): The key (path) of the file.

        Returns:
            dict: A dictionary containing the file metadata.
        """
        key = self._normalize_key(key)
        file_path = Path(self.store.root_path) / key
        if not file_path.exists():
            return {}
        mime_type, _ = mimetypes.guess_type(file_path)
        metadata_store = self._load_metadata_store()
        return {
            "name": key,
            "size": file_path.stat().st_size,
            "modified_time": time.ctime(file_path.stat().st_mtime),
            "created_time": time.ctime(file_path.stat().st_ctime),
            "path": str(file_path),
            "mime_type": mime_type,
            "custom_metadata": metadata_store.get(key, {}).get("custom_metadata", {})
        }

    def save_directory_metadata(self, directory: str, custom_metadata: Dict[str, Any]) -> None:
        """
        Save metadata for a directory.

        Args:
            directory (str): The directory path.
            custom_metadata (Dict[str, Any]): Custom metadata provided by the user.
        """
        metadata_store = self._load_metadata_store()
        metadata_store[directory] = custom_metadata
        self._save_metadata_store(metadata_store)

    def update_directory_metadata(
        self, directory: str, custom_metadata: Dict[str, Any]
    ) -> None:
        """
        Aggiorna (merge) i metadati di una directory.
        Se la directory non ha metadati, li crea da zero.
        """
        metadata_store = self._load_metadata_store()
        existing = metadata_store.get(directory, {})
        if not isinstance(existing, dict):
            existing = {}

        # Merge (new keys overwrite old ones)
        merged = {**existing, **custom_metadata}
        metadata_store[directory] = merged
        self._save_metadata_store(metadata_store)

    def get_directory_metadata(self, directory: str) -> dict:
        """
        Retrieve metadata for a specific directory.

        Args:
            directory (str): The directory path.

        Returns:
            dict: A dictionary containing the directory metadata.
        """
        metadata_store = self._load_metadata_store()
        return metadata_store.get(directory, {})

    def get_directory_metadata_bulk(self, directories: List[str]) -> Dict[str, dict]:
        store = self._get_metadata_store_cached()
        return {d: store.get(d, {}) for d in directories}

    def list_directories(self) -> List[str]:
        """
        Elenca *solo* le directory registrate nel metadata store,
        indipendentemente dal fatto che contengano file o meno.
        """
        metadata_store = self._load_metadata_store()

        # Filtra i record che NON hanno la chiave "size" (=> sono directory)
        directories = [
            key
            for key, meta in metadata_store.items()
            if isinstance(meta, dict) and "size" not in meta
        ]

        return sorted(directories)

    def delete_directory_metadata(self, directory: str) -> None:
        """
        Delete metadata for a specific directory.

        Args:
            directory (str): The directory path.

        Raises:
            KeyError: If the directory metadata does not exist.
        """
        metadata_store = self._load_metadata_store()
        if directory in metadata_store:
            del metadata_store[directory]
            self._save_metadata_store(metadata_store)
        else:
            raise KeyError(f"Metadata for directory '{directory}' does not exist.")

    def search_files(self, query: str, directory: Optional[str] = None) -> List[str]:
        """
        Search for files matching the query within a specific directory or globally.

        Args:
            query (str): The search query.
            directory (Optional[str]): The directory to search within. If None, searches globally.

        Returns:
            List[str]: A list of file keys (paths) matching the query.
        """
        metadata_store = self._load_metadata_store()
        results = []
        for key, metadata in metadata_store.items():
            if directory and not key.startswith(directory):
                continue
            if query.lower() in key.lower() or any(query.lower() in str(value).lower() for value in metadata.values()):
                results.append(key)
        return results

    def filter_files(self, mime_type: Optional[str] = None, min_size: Optional[int] = None,
                     max_size: Optional[int] = None) -> List[str]:
        """
        Filter files based on MIME type and size.

        Args:
            mime_type (Optional[str]): The MIME type to filter by.
            min_size (Optional[int]): The minimum file size in bytes.
            max_size (Optional[int]): The maximum file size in bytes.

        Returns:
            List[str]: A list of file keys (paths) that match the filters.
        """
        metadata_store = self._load_metadata_store()
        results = []
        for key, metadata in metadata_store.items():
            if mime_type and metadata.get("mime_type") != mime_type:
                continue
            size = metadata.get("size")
            if size is not None:
                if min_size is not None and size < min_size:
                    continue
                if max_size is not None and size > max_size:
                    continue
            results.append(key)
        return results
