import concurrent
import logging
import random
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Type, Union

from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
#from langchain_community.document_loaders.unstructured import UnstructuredFileLoader
from langchain_unstructured import UnstructuredLoader

from document_loaders.utilities.pymupdf4llm_loader import PyMuPDFLoader
from document_loaders.utilities.video2text_llm_loader import VideoDescriptionLoader
from document_loaders.utilities.image2text_llm_loader import ImageDescriptionLoader

#from langchain_community.document_loaders.pdf import PyPDFLoader, PyMuPDFLoader

FILE_LOADER_TYPE = Union[
   Type[UnstructuredLoader], Type[TextLoader], Type[BSHTMLLoader], Type[CSVLoader], Type[PyMuPDFLoader], Type[ImageDescriptionLoader], Type[VideoDescriptionLoader]]
logger = logging.getLogger(__name__)


class CustomDirectoryLoader(BaseLoader):
    """
    CustomDirectoryLoader is a versatile document loader that extracts documents from files
    contained in directories. It extends the BaseLoader class from LangChain and supports
    loading documents using different loaders based on file patterns (globs).

    Attributes:
        path (str): Path to the directory.
        loader_map (Dict[str, FILE_LOADER_TYPE]): Mapping of file patterns (globs) to loader classes.
        loader_kwargs_map (Optional[Dict[str, dict]]): Mapping of file patterns (globs) to keyword arguments for loaders.
        metadata_map (Optional[Dict[str, Dict[str, Any]]]): Mapping of file patterns (globs) to metadata dictionaries.
        default_metadata (Optional[Dict[str, Any]]): Default metadata to apply to all documents.
        recursive (bool): Whether to search directories recursively. Defaults to False.
        max_depth (Optional[int]): Maximum depth of directory traversal. If None, no limit is applied.
        silent_errors (bool): Whether to ignore errors silently. Defaults to False.
        load_hidden (bool): Whether to load hidden files. Defaults to False.
        show_progress (bool): Whether to show a progress bar during loading. Defaults to False.
        use_multithreading (bool): Whether to use multithreading for loading. Defaults to False.
        max_concurrency (int): Maximum number of concurrent threads. Defaults to 4.
        exclude (Union[Sequence[str], str]): Patterns to exclude from loading.
        sample_size (int): Maximum number of files to load. If 0, no limit is applied.
        randomize_sample (bool): Whether to randomize the sample of files loaded.
        sample_seed (Union[int, None]): Seed for randomizing the sample of files.
        output_store_map (Optional[Dict[str, Dict[str, Any]]]): Mapping of file patterns (globs) to output storage configurations.
        default_output_store (Optional[Dict[str, Any]]): Default output storage configuration for documents not matching any specific pattern.
    """

    def __init__(
        self,
        path: str,
        loader_map: Dict[str, FILE_LOADER_TYPE],
        loader_kwargs_map: Optional[Dict[str, Dict[str, Any]]] = None,
        metadata_map: Optional[Dict[str, Dict[str, Any]]] = None,
        default_metadata: Optional[Dict[str, Any]] = None,
        recursive: bool = False,
        max_depth: Optional[int] = None,
        silent_errors: bool = False,
        load_hidden: bool = False,
        show_progress: bool = False,
        use_multithreading: bool = False,
        max_concurrency: int = 4,
        exclude: Union[Sequence[str], str] = (),
        sample_size: int = 0,
        randomize_sample: bool = False,
        sample_seed: Union[int, None] = None,
        output_store_map: Optional[Dict[str, Dict[str, Any]]] = None,
        default_output_store: Optional[Dict[str, Any]] = None,
    ):
        """
        Initialize the CustomDirectoryLoader with the specified parameters.

        Args:
            path (str): Path to the directory.
            loader_map (Dict[str, FILE_LOADER_TYPE]): Mapping of file patterns (globs) to loader classes.
            loader_kwargs_map (Optional[Dict[str, dict]]): Mapping of file patterns (globs) to keyword arguments for loaders.
            metadata_map (Optional[Dict[str, Dict[str, Any]]]): Mapping of file patterns (globs) to metadata dictionaries.
            default_metadata (Optional[Dict[str, Any]]): Default metadata to apply to all documents.
            recursive (bool): Whether to search directories recursively. Defaults to False.
            max_depth (Optional[int]): Maximum depth of directory traversal. If None, no limit is applied.
            silent_errors (bool): Whether to ignore errors silently. Defaults to False.
            load_hidden (bool): Whether to load hidden files. Defaults to False.
            show_progress (bool): Whether to show a progress bar during loading. Defaults to False.
            use_multithreading (bool): Whether to use multithreading for loading. Defaults to False.
            max_concurrency (int): Maximum number of concurrent threads. Defaults to 4.
            exclude (Union[Sequence[str], str]): Patterns to exclude from loading.
            sample_size (int): Maximum number of files to load. If 0, no limit is applied.
            randomize_sample (bool): Whether to randomize the sample of files loaded.
            sample_seed (Union[int, None]): Seed for randomizing the sample of files.
            output_store_map (Optional[Dict[str, Dict[str, Any]]]): Mapping of file patterns (globs) to output storage configurations.
            default_output_store (Optional[Dict[str, Any]]): Default output storage configuration for documents not matching any specific pattern.
        """
        self.path = path
        self.loader_map = loader_map
        self.loader_kwargs_map = loader_kwargs_map or {}
        self.metadata_map = metadata_map or {}
        self.default_metadata = default_metadata or {}
        self.recursive = recursive
        self.max_depth = max_depth
        self.silent_errors = silent_errors
        self.load_hidden = load_hidden
        self.show_progress = show_progress
        self.use_multithreading = use_multithreading
        self.max_concurrency = max_concurrency
        self.exclude = exclude
        self.sample_size = sample_size
        self.randomize_sample = randomize_sample
        self.sample_seed = sample_seed
        self.output_store_map = output_store_map or {}
        self.default_output_store = default_output_store or {}

    def load(self) -> List[Document]:
        """
        Load documents from the directory and return them as a list.

        Returns:
            List[Document]: List of loaded documents.
        """
        return list(self.lazy_load())

    def lazy_load(self) -> Iterator[Document]:
        """
        Lazily load documents from the directory.

        Returns:
            Iterator[Document]: An iterator over the loaded documents.
        """
        p = Path(self.path)
        if not p.exists():
            raise FileNotFoundError(f"Directory not found: '{self.path}'")
        if not p.is_dir():
            raise ValueError(f"Expected directory, got file: '{self.path}'")

        paths = p.rglob("**/*") if self.recursive else p.glob("*")
        items = [
            path for path in paths
            if not (self.exclude and any(path.match(glob) for glob in self.exclude))
            and path.is_file() and (self.load_hidden or not path.name.startswith('.'))
            and (self.max_depth is None or len(path.relative_to(p).parts) <= self.max_depth)
        ]

        if self.sample_size > 0:
            if self.randomize_sample:
                randomizer = random.Random(self.sample_seed if self.sample_seed else None)
                randomizer.shuffle(items)
            items = items[:min(len(items), self.sample_size)]

        pbar = None
        if self.show_progress:
            try:
                from tqdm import tqdm
                pbar = tqdm(total=len(items))
            except ImportError:
                logger.warning("To log the progress of CustomDirectoryLoader you need to install tqdm")

        if self.use_multithreading:
            futures = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_concurrency) as executor:
                for i in items:
                    futures.append(
                        executor.submit(self._lazy_load_file_to_non_generator(self._lazy_load_file), i, p, pbar)
                    )
                for future in concurrent.futures.as_completed(futures):
                    for item in future.result():
                        yield item
        else:
            for i in items:
                yield from self._lazy_load_file(i, p, pbar)

        if pbar:
            pbar.close()

    def _lazy_load_file_to_non_generator(self, func: Callable) -> Callable:
        """
        Convert a generator function to a non-generator function.

        Args:
            func (Callable): The generator function to convert.

        Returns:
            Callable: A non-generator function.
        """
        def non_generator(item: Path, path: Path, pbar: Optional[Any]) -> List:
            return [x for x in func(item, path, pbar)]
        return non_generator

    def _lazy_load_file(self, item: Path, path: Path, pbar: Optional[Any]) -> Iterator[Document]:
        """
        Load a file and yield its documents.

        Args:
            item (Path): The file path.
            path (Path): The directory path.
            pbar (Optional[Any]): The progress bar object.

        Yields:
            Iterator[Document]: An iterator over the documents in the file.
        """
        if item.is_file():
            for pattern, loader_cls in self.loader_map.items():
                if item.match(pattern):
                    try:
                        loader_kwargs = self.loader_kwargs_map.get(pattern, {})
                        loader = loader_cls(str(item), **loader_kwargs)
                        for subdoc in loader.lazy_load():
                            # Apply specific and default metadata
                            specific_metadata = self.metadata_map.get(pattern, {})
                            subdoc.metadata.update(self.default_metadata)
                            subdoc.metadata.update(specific_metadata)
                            yield subdoc
                    except Exception as e:
                        if self.silent_errors:
                            logger.warning(f"Error loading file {str(item)}: {e}")
                        else:
                            raise e
                    finally:
                        if pbar:
                            pbar.update(1)


'''if __name__ == "__main__":
   # from document_loaders.utilities.image2text_llm_loader import ImageDescriptionLoader

    loader_map = {
        "*.pdf": UnstructuredFileLoader,
        #"img.jpg": ImageDescriptionLoader,  # Add support for images
        #"*.png": ImageDescriptionLoader,  # Add PNG support
    }

    loader_kwargs_map = {
        "*.pdf": {
            #"pages": None,
            #"page_chunks": True,
            #"write_images": True,
            #"image_size_limit": 0.025,
            #"embed_images": True,
            #"image_path": "C:\\Users\\Golden Bit\\Desktop\\projects_in_progress\\GoldenProjects\\golden_bit\\repositories\\nlp-core-api\\tmp",
            "chunking_strategy": "by_page",
            "strategy": "hi_res",
            "partition_via_api": False,
        },  # Add arguments for PyMuPDFLoader
        "img.jpg": {"openai_api_key": "....", "resize_to": (256, 256)},
        # Arguments for ImageDescriptionLoader
        "*.png": {"openai_api_key": "....", "resize_to": (256, 256)},
        # Arguments for ImageDescriptionLoader
    }

    metadata_map = {
        "*.pdf": {"type": "document", "source": "PDF files"},
        "img.jpg": {"type": "image", "source": "Image files"},
        "*.png": {"type": "image", "source": "Image files"},
    }

    loader = CustomDirectoryLoader(
        path="C:\\Users\\info\\Desktop\\work_space\\repositories\\nlp-core-api\\data\\my_data",
        loader_map=loader_map,
        loader_kwargs_map=loader_kwargs_map,
        metadata_map=metadata_map,
        recursive=True,
        max_depth=3,
        show_progress=True,  # Show progress bar
    )

    documents = loader.load()

    for doc in documents:
        print(f"File: {doc.metadata.get('source')}, Type: {doc.metadata.get('type')}")
        print(f"Content: {doc.page_content}")
        print(f"Metadata: {doc.metadata}")
        print("\n---\n")'''

if __name__ == "__main__":
    # === CONFIG API SELF-HOSTED ===
    API_URL = "http://localhost:8000/"  # endpoint completo
    API_KEY = "metti-una-chiave-robusta"                  # deve combaciare col container

    # Directory da scandire
    ROOT_DIR = r"C:\Users\info\Desktop\work_space\repositories\nlp-core-api\data\my_data"

    # Modalità post-processing del loader: "elements" | "single" | "paged"
    MODE = "elements"

    # Parametri passati all'API di Unstructured (valori FISSI nello script)
    UNSTRUCTURED_KWARGS = {
        # --- strategia / modelli ---
        "strategy": "hi_res",                   # "fast" | "hi_res" | "ocr_only" | "auto"
        "hi_res_model_name": "yolox",           # o "detectron2_onnx" se disponibile
        "output_format": "application/json",    # con LangChain è consigliato JSON

        # --- OCR / lingue ---
        "ocr_languages": ["ita", "eng"],        # codici Tesseract (ita, eng, fra, deu, spa, ...)
        "languages": ["it", "en"],              # hint ISO-639-1 del testo
        "encoding": "utf-8",

        # --- layout / coordinate / pagination ---
        "coordinates": True,
        "include_page_breaks": True,
        "starting_page_number": 1,

        # --- tabelle PDF ---
        "pdf_infer_table_structure": True,      # solo con "hi_res"/"auto"
        "skip_infer_table_types": [],           # es. ["pdf"] per saltare

        # --- XML/HTML ---
        "xml_keep_tags": False,

        # --- slide ---
        "include_slide_notes": True,

        # --- chunking (pre-Document split di Unstructured) ---
        "chunking_strategy": "by_title",        # "basic" | "by_title" | None (evita "by_page": non esiste qui)
        "combine_under_n_chars": 2000,
        "max_characters": 4000,
        "multipage_sections": True,
        "new_after_n_chars": None,              # None = disabilitato
        "overlap": 200,
        "overlap_all": False,
        "unique_element_ids": True,
    }

    # Mappa estensioni → loader
    # Per i PDF usiamo il NUOVO loader collegato all'API self-hosted
    loader_map = {
        "*.pdf": UnstructuredLoader,  # <--- NUOVO
        # aggiungi altri pattern se vuoi (txt/html/csv/immagini ecc.)
        "*.txt": TextLoader,
        "*.html": BSHTMLLoader,
        "*.htm": BSHTMLLoader,
        "*.csv": CSVLoader,
        # Esempio: fallback alternativo PyMuPDF per PDF molto problematici
        # "*.pdf_fallback": PyMuPDFLoader,
    }

    # Parametri specifici per ogni loader/pattern
    loader_kwargs_map = {
        "*.pdf": {
            "mode": MODE,
            "partition_via_api": True,      # usa l'API (non il processing locale)
            "url": API_URL,                 # endpoint completo
            "api_key": API_KEY,             # header 'unstructured-api-key'
            **UNSTRUCTURED_KWARGS,          # tutti i parametri fissati sopra
        },
        "*.txt": {
            "encoding": "utf-8",
        },
        "*.html": {
            "open_encoding": "utf-8",
        },
        "*.htm": {
            "open_encoding": "utf-8",
        },
        "*.csv": {
            "encoding": "utf-8",
        },
        # Esempio fallback PyMuPDF:
        # "*.pdf_fallback": {
        #     "text_kwargs": {"page_numbers": None},
        # },
    }

    # Metadati da applicare per pattern (verranno mergiati con i default)
    metadata_map = {
        "*.pdf": {"type": "document", "source": "PDF via Unstructured API"},
        "*.txt": {"type": "text", "source": "TXT files"},
        "*.html": {"type": "html", "source": "HTML files"},
        "*.htm": {"type": "html", "source": "HTML files"},
        "*.csv": {"type": "table", "source": "CSV files"},
    }

    # Metadati di default per tutti
    default_metadata = {
        "loader": "CustomDirectoryLoader",
        "project": "nlp-core-api",
    }

    loader = CustomDirectoryLoader(
        path=ROOT_DIR,
        loader_map=loader_map,
        loader_kwargs_map=loader_kwargs_map,
        metadata_map=metadata_map,
        default_metadata=default_metadata,
        recursive=True,
        max_depth=3,
        show_progress=True,
        use_multithreading=True,   # opzionale: parallelo lato client
        max_concurrency=4,
        # exclude=("**/*.tmp", "**/~$*"),  # esempi di esclusioni
        # sample_size=100, randomize_sample=True, sample_seed=42,
        silent_errors=False,       # metti True per non fermarti al primo errore
    )

    documents = loader.load()

    # Stampa un piccolo riepilogo
    print(f"\nTotale documenti caricati: {len(documents)}")
    for i, doc in enumerate(documents[:5], start=1):
        print(f"\n--- DOC {i} ---")
        print(doc.page_content[:400])
        print(doc.metadata)
