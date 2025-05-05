import base64
import os
from fastapi import FastAPI, HTTPException, Form, Query, Path, Body, APIRouter, BackgroundTasks
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any
import json
import uuid
from pymongo import MongoClient
from langchain_core.documents import Document
from langchain_unstructured import UnstructuredLoader
from document_loaders.utilities.image2text_llm_loader import ImageDescriptionLoader
#from langchain_community.document_loaders import PyMuPDFLoader
from document_loaders.utilities.pymupdf4llm_loader import PyMuPDFLoader
from langchain_community.document_loaders.csv_loader import CSVLoader
from langchain_community.document_loaders.html_bs import BSHTMLLoader
from langchain_community.document_loaders.text import TextLoader
from document_loaders.utilities.custom_directory_loader import CustomDirectoryLoader
from document_loaders.utilities.video2text_llm_loader import VideoDescriptionLoader
from datetime import datetime

router = APIRouter()

########################################################################################################################
# MongoDB connection configuration
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING', 'localhost')
mongo_client = MongoClient(MONGO_CONNECTION_STRING)
loaders_db_name = "loader_configs"
document_store_db_name = "document_store"
########################################################################################################################

# Mappatura dei loader disponibili
available_loaders = {
    "TextLoader": TextLoader,
    "BSHTMLLoader": BSHTMLLoader,
    "CSVLoader": CSVLoader,
    "PyMuPDFLoader": PyMuPDFLoader,
    "ImageDescriptionLoader": ImageDescriptionLoader,
    "UnstructuredLoader": UnstructuredLoader,
    "VideoDescriptionLoader": VideoDescriptionLoader
}


########################################################################################################################
# ---------- TASK STATE HELPERS ----------

def _create_task_record(task_id: str,
                        endpoint: str,
                        payload: dict) -> None:
    """Registra un job: id (nostro), endpoint e input JSON."""
    res = mongo_client[loaders_db_name]["tasks"].insert_one({
        "id": task_id,
        "endpoint": endpoint,
        "payload": payload,
        "status": "PENDING",
        "created_at": datetime.utcnow(),
        "finished_at": None,
        "result": None,
        "error": None
    })


def _update_task_status(task_id: str,
                        status: str,
                        *,
                        result: Optional[List[dict]] = None,
                        error: Optional[str] = None) -> None:
    res = mongo_client[loaders_db_name].tasks.update_one(
        {"id": task_id},                 # <-- ora filtra su `id`
        {"$set": {
            "status": status,
            "finished_at": (
                datetime.utcnow() if status in ("DONE", "ERROR") else None
            ),
            "result": result,
            "error": error
        }}
    )


########################################################################################################################
# ---------------- BACKGROUND WORKERS ------------

def _process_loader_job(config_id: str, task_id: str) -> None:
    """
    Esegue in background tutta la logica di load_documents.
    Aggiorna lo stato del job in Mongo.
    """
    try:
        _update_task_status(task_id, "RUNNING")

        config_doc = mongo_client[loaders_db_name].configs.find_one({"_id": config_id})
        if not config_doc:
            raise RuntimeError("Configuration not found")

        cfg = config_doc["config"]
        # risolvi le classi dei loader
        cfg["loader_map"] = {g: available_loaders[l] for g, l in cfg["loader_map"].items()}
        cfg.pop("config_id", None)

        loader = CustomDirectoryLoader(**cfg)
        documents = loader.load()
        doc_models = [DocumentModel.from_langchain_document(d) for d in documents]

        # persistenza documenti (identico a prima)
        for dm in doc_models:
            matched = False
            if loader.output_store_map:
                for glob, store_cfg in loader.output_store_map.items():
                    if glob in dm.metadata.get("source", ""):
                        save_document_to_store(store_cfg["collection_name"], dm)
                        matched = True
                        break
            if not matched and loader.default_output_store:
                save_document_to_store(loader.default_output_store["collection_name"], dm)

        _update_task_status(
            task_id, "DONE",
            result=[dm.model_dump() for dm in doc_models]
        )

    except Exception as exc:        # pragma: no cover
        _update_task_status(task_id, "ERROR", error=str(exc))

def _process_loader_job_b64(config_id: str, task_id: str, b64_docs: List[str]) -> None:
    """
    Variante che crea una dir temporanea, salva i file b64 e poi
    richiama lo stesso flusso del loader.
    """
    try:
        _update_task_status(task_id, "RUNNING")

        # 1. crea dir temporanea
        temp_dir = f"/tmp/{uuid.uuid4()}"
        os.makedirs(temp_dir, exist_ok=True)

        # 2. scrive i file
        for doc_b64 in b64_docs:
            tmp_name = f"{uuid.uuid4()}.pdf"   # o deduci estensione dal client
            with open(os.path.join(temp_dir, tmp_name), "wb") as fh:
                fh.write(base64.b64decode(doc_b64))

        # 3. carica config da Mongo
        cfg_doc = mongo_client[loaders_db_name].configs.find_one({"_id": config_id})
        if not cfg_doc:
            raise RuntimeError("Configuration not found")
        cfg = cfg_doc["config"]
        cfg["path"] = temp_dir
        cfg["loader_map"] = {g: available_loaders[l] for g, l in cfg["loader_map"].items()}
        cfg.pop("config_id", None)

        loader = CustomDirectoryLoader(**cfg)
        documents = loader.load()
        doc_models = [DocumentModel.from_langchain_document(d) for d in documents]

        # stessa persistenza
        for dm in doc_models:
            matched = False
            if loader.output_store_map:
                for glob, store_cfg in loader.output_store_map.items():
                    if glob in dm.metadata.get("source", ""):
                        save_document_to_store(store_cfg["collection_name"], dm)
                        matched = True
                        break
            if not matched and loader.default_output_store:
                save_document_to_store(loader.default_output_store["collection_name"], dm)

        _update_task_status(
            task_id, "DONE",
            result=[dm.model_dump() for dm in doc_models]
        )

    except Exception as exc:
        _update_task_status(task_id, "ERROR", error=str(exc))

########################################################################################################################


class TaskInfo(BaseModel):
    id: str
    status: str
    endpoint: Optional[str] = None
    payload: Optional[dict] = None
    result: Optional[Any] = None
    error: Optional[str] = None


class LoaderConfig(BaseModel):
    """
    Pydantic model for loader configuration.
    """
    config_id: Optional[str] = Field(None, description="The unique ID for the loader configuration.",
                                     example="abcd1234-efgh-5678-ijkl-9012mnop3456")
    path: str = Field("/path/to/documents", description="The path to the directory containing documents to load.",
                      example="/path/to/documents")
    loader_map: Dict[str, str] = Field({"*.txt": "TextLoader", "*.html": "BSHTMLLoader"},
                                       description="Mapping of file patterns to loader classes.",
                                       example={"*.txt": "TextLoader", "*.html": "BSHTMLLoader"})
    loader_kwargs_map: Optional[Dict[str, dict]] = Field({"*.csv": {"delimiter": ","}},
                                                         description="Optional keyword arguments for each loader.",
                                                         example={"*.csv": {"delimiter": ","}})
    metadata_map: Optional[Dict[str, Dict[str, Any]]] = Field({"*.txt": {"author": "John Doe"}},
                                                              description="Mapping of file patterns to metadata.",
                                                              example={"*.txt": {"author": "John Doe"}})
    default_metadata: Optional[Dict[str, Any]] = Field({"project": "AI Research"},
                                                       description="Default metadata to apply to all documents.",
                                                       example={"project": "AI Research"})
    recursive: bool = Field(True, description="Whether to load documents recursively from subdirectories.",
                            example=True)
    max_depth: Optional[int] = Field(5, description="Maximum depth for recursive loading.", example=5)
    silent_errors: bool = Field(True, description="Whether to ignore errors during document loading.", example=True)
    load_hidden: bool = Field(True, description="Whether to load hidden files.", example=True)
    show_progress: bool = Field(True, description="Whether to show progress during loading.", example=True)
    use_multithreading: bool = Field(True, description="Whether to use multithreading for loading.", example=True)
    max_concurrency: int = Field(8, description="Maximum number of concurrent threads for loading.", example=8)
    exclude: Optional[List[str]] = Field(["*.tmp", "*.log"],
                                         description="List of file patterns to exclude from loading.",
                                         example=["*.tmp", "*.log"])
    sample_size: int = Field(10, description="Number of samples to load. If 0, all documents are loaded.", example=10)
    randomize_sample: bool = Field(True, description="Whether to randomize the sample.", example=True)
    sample_seed: Optional[int] = Field(42, description="Seed for randomization of the sample.", example=42)
    output_store_map: Optional[Dict[str, Dict[str, Any]]] = Field({"*.txt": {"collection_name": "text_files"}},
                                                                  description="Mapping of file patterns to output stores.",
                                                                  example={"*.txt": {"collection_name": "text_files"}})
    default_output_store: Optional[Dict[str, Any]] = Field({"collection_name": "default_store"},
                                                           description="Default output store configuration.",
                                                           example={"collection_name": "default_store"})


class DocumentModel(BaseModel):
    """
    Pydantic model for documents.
    """
    page_content: str = Field(
        "This is the content of the document.",
        description="The content of the document.",
        example="This is the content of the document.")
    metadata: Dict[str, Any] = Field(
        {"author": "John Doe", "category": "Research"},
        description="Metadata associated with the document.",
        example={"author": "John Doe", "category": "Research"})
    type: str = Field(
        "Document",
        description="Type of the document.",
        example="Document")

    @staticmethod
    def from_langchain_document(doc: Document) -> 'DocumentModel':
        """Create a DocumentModel from a LangChain Document."""
        return DocumentModel(
            page_content=doc.page_content,
            metadata=doc.metadata,
            type=doc.type
        )

    def to_langchain_document(self) -> Document:
        """Convert DocumentModel to a LangChain Document."""
        return Document(
            page_content=self.page_content,
            metadata=self.metadata,
        )


class SearchConfig(BaseModel):
    """
    Pydantic model for searching configurations.
    """
    path: Optional[str] = Field("/path/to/documents", description="Path to filter configurations.",
                                example="/path/to/documents")
    loader_type: Optional[str] = Field("TextLoader", description="Loader type to filter configurations.",
                                       example="TextLoader")
    recursive: Optional[bool] = Field(True, description="Whether to filter by recursive loading.", example=True)
    max_depth: Optional[int] = Field(5, description="Maximum depth to filter configurations.", example=5)


# In-memory storage for configurations (for demonstration purposes)
loader_configs = {}


def save_document_to_store(collection_name: str, document: DocumentModel):
    """
    Save a document to the MongoDB document store.

    This function saves the provided document to the specified MongoDB collection. If the document
    already exists (based on the 'id' in metadata), it is updated; otherwise, a new entry is created.

    Args:
        collection_name (str): The name of the collection.
        document (DocumentModel): The document to save.

    Returns:
        str: A unique ID linked to the saved document.
    """
    collection = mongo_client[document_store_db_name][collection_name]
    key = document.metadata.get("id", str(uuid.uuid4()))
    document.metadata.update({"doc_store_id": key, "doc_store_collection": collection_name})
    collection.update_one(
        {"_id": key}, {"$set": {
            "value": {"page_content": document.page_content, "metadata": document.metadata, "type": document.type}}},
        upsert=True
    )

    return key


@router.post("/configure_loader", response_model=str)
async def configure_loader(
        config_id: Optional[str] = Body(None,
                                        description="The unique ID for the loader configuration. If not provided, a new ID will be generated.",
                                        example="abcd1234-efgh-5678-ijkl-9012mnop3456"),
        path: str = Body(
            "/path/to/documents",
            description="The path to the directory containing documents to load.",
            example="/path/to/documents"),
        loader_map: Dict[str, str] = Body(
            {"*.txt": "TextLoader", "*.html": "BSHTMLLoader"},
            description="Mapping of file patterns to loader classes.",
            example={"*.txt": "TextLoader", "*.html": "BSHTMLLoader"}),
        loader_kwargs_map: Optional[Dict[str, dict]] = Body(
            {"*.csv": {"delimiter": ","}},
            description="Optional keyword arguments for each loader.",
            example={"*.csv": {"delimiter": ","}}),
        metadata_map: Optional[Dict[str, Dict[str, Any]]] = Body(
            {"*.txt": {"author": "John Doe"}},
            description="Mapping of file patterns to metadata.",
            example={"*.txt": {"author": "John Doe"}}),
        default_metadata: Optional[Dict[str, Any]] = Body(
            {"project": "AI Research"},
            description="Default metadata to apply to all documents.",
            example={"project": "AI Research"}),
        recursive: bool = Body(
            True,
            description="Whether to load documents recursively from subdirectories.",
            example=True),
        max_depth: Optional[int] = Body(
            5,
            description="Maximum depth for recursive loading.",
            example=5),
        silent_errors: bool = Body(
            True,
            description="Whether to ignore errors during document loading.",
            example=True),
        load_hidden: bool = Body(
            True,
            description="Whether to load hidden files.",
            example=True),
        show_progress: bool = Body(
            True,
            description="Whether to show progress during loading.",
            example=True),
        use_multithreading: bool = Body(
            True,
            description="Whether to use multithreading for loading.",
            example=True),
        max_concurrency: int = Body(
            8,
            description="Maximum number of concurrent threads for loading.",
            example=8),
        exclude: Optional[List[str]] = Body(
            ["*.tmp", "*.log"],
            description="List of file patterns to exclude from loading.",
            example=["*.tmp", "*.log"]),
        sample_size: int = Body(
            10,
            description="Number of samples to load. If 0, all documents are loaded.",
            example=10),
        randomize_sample: bool = Body(
            True,
            description="Whether to randomize the sample.",
            example=True),
        sample_seed: Optional[int] = Body(
            42,
            description="Seed for randomization of the sample.",
            example=42),
        output_store_map: Optional[Dict[str, Dict[str, Any]]] = Body(
            {"*.txt": {"collection_name": "text_files"}},
            description="Mapping of file patterns to output stores.",
            example={"*.txt": {"collection_name": "text_files"}}),
        default_output_store: Optional[Dict[str, Any]] = Body(
            {"collection_name": "default_store"},
            description="Default output store configuration.",
            example={"collection_name": "default_store"})
):
    """
    Configure a CustomDirectoryLoader with specified parameters.

    This endpoint allows the user to set up a CustomDirectoryLoader by specifying
    various parameters including the path, loader mapping, loader arguments, metadata,
    and other optional settings. It returns a unique configuration ID for the loader.

    If `config_id` is provided, the endpoint will check if a configuration with the same ID already exists.
    If it does, an error will be returned.

    This endpoint is useful for setting up document loading configurations dynamically.
    The configurations are stored in MongoDB and can be retrieved using their unique IDs.
    """
    try:
        if config_id and mongo_client[loaders_db_name].configs.find_one({"_id": config_id}):
            raise HTTPException(status_code=400, detail=f"Configuration with ID {config_id} already exists.")

        # Generate a new config_id if not provided
        if not config_id:
            config_id = str(uuid.uuid4())

        # Convert string representation to actual class objects
        resolved_loader_map = {glob: available_loaders[loader] for glob, loader in loader_map.items()}

        loader_configs[config_id] = CustomDirectoryLoader(
            path=path,
            loader_map=resolved_loader_map,
            loader_kwargs_map=loader_kwargs_map,
            metadata_map=metadata_map,
            default_metadata=default_metadata,
            recursive=recursive,
            max_depth=max_depth,
            silent_errors=silent_errors,
            load_hidden=load_hidden,
            show_progress=show_progress,
            use_multithreading=use_multithreading,
            max_concurrency=max_concurrency,
            exclude=exclude,
            sample_size=sample_size,
            randomize_sample=randomize_sample,
            sample_seed=sample_seed,
            output_store_map=output_store_map,
            default_output_store=default_output_store
        )

        # Save configuration to MongoDB
        mongo_client[loaders_db_name].configs.insert_one({
            "_id": config_id,
            "config": {
                "config_id": config_id,
                "path": path,
                "loader_map": loader_map,
                "loader_kwargs_map": loader_kwargs_map,
                "metadata_map": metadata_map,
                "default_metadata": default_metadata,
                "recursive": recursive,
                "max_depth": max_depth,
                "silent_errors": silent_errors,
                "load_hidden": load_hidden,
                "show_progress": show_progress,
                "use_multithreading": use_multithreading,
                "max_concurrency": max_concurrency,
                "exclude": exclude,
                "sample_size": sample_size,
                "randomize_sample": randomize_sample,
                "sample_seed": sample_seed,
                "output_store_map": output_store_map,
                "default_output_store": default_output_store
            }
        })

        return config_id
    except KeyError as e:
        raise HTTPException(status_code=400, detail=f"Loader {str(e)} not found in available loaders.")
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/load_documents/{config_id}", response_model=List[DocumentModel])
async def load_documents(
        config_id: str = Path(
            ...,
            description="The unique ID of the loader configuration.",
            example="abcd1234-efgh-5678-ijkl-9012mnop3456")
):
    """
    Load documents using a pre-configured loader.

    This endpoint loads documents from a directory using a pre-configured loader,
    identified by a unique configuration ID. The documents are returned along with
    their metadata. If configured, documents are also stored in the MongoDB document store.

    This endpoint is designed to provide a flexible document loading mechanism where
    the configuration can be set up once and reused multiple times for different document
    loading tasks.
    """
    #if config_id not in loader_configs:
    #    raise HTTPException(status_code=404, detail="Configuration not found")

    config = mongo_client[loaders_db_name].configs.find_one({"_id": config_id})
    config = config["config"] if "config" in config else None
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    # Convert string representation to actual class objects
    resolved_loader_map = {glob: available_loaders[loader] for glob, loader in config["loader_map"].items()}

    config["loader_map"] = resolved_loader_map
    del config["config_id"]

    loader_configs[config_id] = CustomDirectoryLoader(**config)

    loader = loader_configs[config_id]
    documents = loader.load()
    document_models = [DocumentModel.from_langchain_document(doc) for doc in documents]

    # Save documents to the document store if configured
    for doc_model in document_models:
        matched = False
        if loader.output_store_map:
            for glob, store_config in loader.output_store_map.items():
                if glob in doc_model.metadata.get("source", ""):
                    collection_name = store_config.get("collection_name")
                    if collection_name:
                        save_document_to_store(collection_name, doc_model)
                        matched = True
                        break
        if not matched and loader.default_output_store:
            collection_name = loader.default_output_store.get("collection_name")
            if collection_name:
                save_document_to_store(collection_name, doc_model)

    return document_models


@router.post("/load_b64_documents/{config_id}", response_model=List[DocumentModel])
async def load_b64_documents(
        config_id: str = Path(
            ...,
            description="The unique ID of the loader configuration.",
            example="abcd1234-efgh-5678-ijkl-9012mnop3456"
        ),
        documents: List[str] = Form(
            ...,
            description="List of files in base64 string format",
            example=["<base64>", "<base64>"]
        )
):
    """
    Load documents using a pre-configured loader.

    This endpoint loads documents from a directory using a pre-configured loader,
    identified by a unique configuration ID. The documents are returned along with
    their metadata. If configured, documents are also stored in the MongoDB document store.

    This endpoint is designed to provide a flexible document loading mechanism where
    the configuration can be set up once and reused multiple times for different document
    loading tasks.
    """

    #if config_id not in loader_configs:
    #    raise HTTPException(status_code=404, detail="Configuration not found")

    # Create a temporary directory using a unique UUID for this session
    temp_dir = f"/tmp/{str(uuid.uuid4())}"
    os.makedirs(temp_dir, exist_ok=True)

    # List to store the document metadata to be returned
    #documents_metadata = []

    for doc_b64 in documents:
        # Generate a unique filename for each document
        filename = f"{str(uuid.uuid4())}.pdf"  # Assuming PDF files, adjust as needed

        # Save the file to the temporary directory
        doc_path = os.path.join(temp_dir, filename)

        # Decode the base64 string and save it to the file
        try:
            with open(doc_path, "wb") as file:
                file.write(base64.b64decode(doc_b64))
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to save document: {str(e)}")

        # Store metadata for the document
        #documents_metadata.append(DocumentModel(filename=filename, path=doc_path))


    ####

    config = mongo_client[loaders_db_name].configs.find_one({"_id": config_id})
    config = config["config"] if "config" in config else None
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")


    # Modify the loader config to point to the new temporary directory
    config["path"] = temp_dir

    # Convert string representation to actual class objects
    resolved_loader_map = {glob: available_loaders[loader] for glob, loader in config["loader_map"].items()}

    config["loader_map"] = resolved_loader_map
    del config["config_id"]

    loader_configs[config_id] = CustomDirectoryLoader(**config)

    loader = loader_configs[config_id]
    documents = loader.load()
    document_models = [DocumentModel.from_langchain_document(doc) for doc in documents]

    # Save documents to the document store if configured
    for doc_model in document_models:
        matched = False
        if loader.output_store_map:
            for glob, store_config in loader.output_store_map.items():
                if glob in doc_model.metadata.get("source", ""):
                    collection_name = store_config.get("collection_name")
                    if collection_name:
                        save_document_to_store(collection_name, doc_model)
                        matched = True
                        break
        if not matched and loader.default_output_store:
            collection_name = loader.default_output_store.get("collection_name")
            if collection_name:
                save_document_to_store(collection_name, doc_model)

    return document_models


@router.get("/get_config/{config_id}", response_model=LoaderConfig)
async def get_config_by_id(
        config_id: str = Path(...,
                              description="The unique ID of the loader configuration to retrieve.",
                              example="abcd1234-efgh-5678-ijkl-9012mnop3456")
):
    """
    Retrieve a single loader configuration by its unique ID.

    This endpoint retrieves a loader configuration stored in MongoDB by its unique ID.
    It is useful for accessing specific configurations without listing all configurations.
    """
    config = mongo_client[loaders_db_name].configs.find_one({"_id": config_id})
    if not config:
        raise HTTPException(status_code=404, detail="Configuration not found")

    return LoaderConfig(**config["config"])


@router.get("/list_configs", response_model=List[LoaderConfig])
async def list_configs():
    """
    List all existing loader configurations.

    This endpoint retrieves and returns all existing loader configurations stored in the MongoDB.
    It is useful for viewing all available configurations.
    """
    configs = mongo_client[loaders_db_name].configs.find()
    return [LoaderConfig(**config["config"]) for config in configs]


@router.post("/search_configs", response_model=List[LoaderConfig])
async def search_configs(search_params: SearchConfig = Body(...)):
    """
    Search for loader configurations based on advanced criteria.

    This endpoint allows the user to search for existing loader configurations based on specified criteria such as
    path, loader type, recursive loading, and maximum depth.
    """
    query = {}
    if search_params.path:
        query["config.path"] = search_params.path
    if search_params.loader_type:
        query["config.loader_map"] = {"$in": [search_params.loader_type]}
    if search_params.recursive is not None:
        query["config.recursive"] = search_params.recursive
    if search_params.max_depth is not None:
        query["config.max_depth"] = search_params.max_depth

    configs = mongo_client[loaders_db_name].configs.find(query)
    return [LoaderConfig(**config["config"]) for config in configs]


@router.delete("/delete_config/{config_id}", response_model=dict)
async def delete_config(
        config_id: str = Path(
            ...,
            description="The unique ID of the loader configuration to delete.",
            example="abcd1234-efgh-5678-ijkl-9012mnop3456")
):
    """
    Delete a loader configuration.

    This endpoint deletes a specified loader configuration from the MongoDB.
    """
    result = mongo_client[loaders_db_name].configs.delete_one({"_id": config_id})
    if result.deleted_count == 0:
        raise HTTPException(status_code=404, detail="Configuration not found")
    return {"detail": "Configuration deleted successfully"}

########################################################################################################################

@router.post("/load_documents_async/{config_id}", response_model=dict)
async def load_documents_async(
    background_tasks: BackgroundTasks,
    config_id: str = Path(..., description="Loader configuration ID"),
    task_id: Optional[str] = Form(None, description="Task ID")
):

    """
    Avvia il caricamento dei documenti in background e
    restituisce subito task_id e stato iniziale.
    """

    task_id = str(uuid.uuid4()) if not task_id else task_id

    _create_task_record(
        task_id=task_id,
        endpoint="load_documents_async",
        payload={"config_id": config_id}
    )

    background_tasks.add_task(
        _process_loader_job,
        config_id=config_id,
        task_id=task_id
    )

    return {"task_id": task_id, "status": "PENDING"}


@router.post("/load_b64_documents_async/{config_id}", response_model=dict)
async def load_b64_documents_async(
    background_tasks: BackgroundTasks,
    config_id: str = Path(..., description="Loader configuration ID"),
    documents: List[str] = Form(..., description="Lista di file in base64"),
    task_id: Optional[str] = Form(None, description="Task ID")
):
    """
    Variante che accetta file base-64: avvia il job in background
    mantenendo identica la firma di input.
    """
    task_id = str(uuid.uuid4()) if not task_id else task_id

    _create_task_record(
        task_id=task_id,
        endpoint="load_b64_documents_async",
        payload={"config_id": config_id, "documents_count": len(documents)}
    )
    background_tasks.add_task(
        _process_loader_job_b64,
        config_id=config_id,
        task_id=task_id,
        b64_docs=documents
    )
    return {"task_id": task_id, "status": "PENDING"}


@router.get("/task_status/{task_id}", response_model=TaskInfo)
async def get_task_status(
    task_id: str = Path(..., description="ID del job restituito alla creazione")
):

    record = mongo_client[loaders_db_name]["tasks"].find_one({"id": task_id})
    if not record:
        raise HTTPException(status_code=404, detail="Task not found")

    docs = None
    if record["status"] == "DONE" and record["result"]:
        docs = [DocumentModel(**d) for d in record["result"]]

    return TaskInfo(
        id=task_id,
        status=record["status"],
        result=docs,
        error=record.get("error")
    )

########################################################################################################################


if __name__ == "__main__":
    import uvicorn

    app = FastAPI()

    app.include_router(router, prefix="/document_loaders", tags=["document_loaders"])

    uvicorn.run(app, host="127.0.0.1", port=8101)
