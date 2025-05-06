"""
vectorstore_toolkit.py

Full rewrite: supports string‑based inputs for metadata filtering and k, suitable
for LangChain tool calls where every argument must be a string. The strings are
parsed (JSON → dict, str → int) with graceful error handling.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any

from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool

# Registry helpers (assumed to wrap a Chroma vector store)
from vector_stores.api import vector_stores, load_vector_store

###############################################################################
# Helper to obtain (and lazily load) the vector store                        #
###############################################################################

def get_vectorstore_component(store_id: str):
    """Return a vector store instance from the global registry, loading it on‑demand."""
    if store_id not in vector_stores:
        # The config id convention is <store_id>_config (change if different)
        load_vector_store(config_id=f"{store_id}_config")
    return vector_stores[store_id]

###############################################################################
# Pydantic schema for the tool                                               #
###############################################################################

class SearchModel(BaseModel):
    """Input schema for the search tool.

    All fields are **strings** to comply with the generic tool‑calling contract.
    They are parsed into Python types inside the search function.
    """

    query: str = Field(
        ...,
        title="Query",
        description="Free‑text query to search for in the vector store.",
    )
    metadata_filter: Optional[str] = Field(
        default="{}",
        title="Metadata filter (JSON string)",
        description=(
            "A JSON‑encoded dictionary specifying Chroma metadata filters. "
            """Example: '{"year": 2024, "author": "Ada"}'. """
            "Supports Mongo‑style operators (e.g. {'price': {'$lt': 50}}) when serialized as JSON."
        ),
    )
    k: Optional[str] = Field(
        default="10",
        title="k (stringified integer)",
        description="String‑encoded integer that overrides the number of top documents to return.",
    )

    class Config:
        # Strict mode: forbid extras so the LLM makes less mistakes
        extra = "forbid"

###############################################################################
# Toolkit manager                                                            #
###############################################################################

class VectorStoreToolKitManager:
    """Bridge between a LangChain vector store (Chroma) and a StructuredTool."""

    def __init__(
        self,
        store_id: str,
        search_type: str = "similarity",
        default_k: int = 10,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.store_id = store_id
        self.vectorstore = get_vectorstore_component(store_id=store_id)
        self.search_type = search_type
        # baseline kwargs used for *every* call; may be overridden per‑call
        self.base_search_kwargs: Dict[str, Any] = search_kwargs or {}
        self.base_search_kwargs.setdefault("k", default_k)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_metadata_filter(filter_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Convert JSON string → dict for metadata filter; raise user‑friendly errors."""
        if not filter_str:
            return None
        try:
            parsed = json.loads(filter_str)
            if not isinstance(parsed, dict):
                raise ValueError("metadata_filter must decode to a JSON object/dict")
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'metadata_filter' JSON: {exc}. "
                "Expected something like '{\"year\": 2024}'."
            ) from exc

    @staticmethod
    def _parse_k(k_str: Optional[str]) -> Optional[int]:
        """Convert k string → int (positive) or None."""
        if k_str in (None, ""):
            return None
        try:
            k_int = int(k_str)
            if k_int <= 0:
                raise ValueError("k must be positive")
            return k_int
        except ValueError as exc:
            raise ValueError(f"Invalid 'k': {exc}. Provide a positive integer as string.") from exc

    def _build_search_kwargs(
        self,
        metadata_filter: Optional[Dict[str, Any]],
        k: Optional[int],
    ) -> Dict[str, Any]:
        """Merge base kwargs with per‑call overrides (filter, k)."""
        kwargs = dict(self.base_search_kwargs)  # shallow copy
        if metadata_filter is not None:
            kwargs["filter"] = metadata_filter
        if k is not None:
            kwargs["k"] = k
        return kwargs

    # ------------------------------------------------------------------ #
    # Core retrieval logic
    # ------------------------------------------------------------------ #

    def _get_retriever(
        self,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
    ):
        search_kwargs = self._build_search_kwargs(metadata_filter, k)
        return self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Public method to expose via the tool
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        metadata_filter: Optional[str] = None,
        k: Optional[str] = None,
    ) -> Any:
        """Search the vector store.

        Parameters are strings because LangChain passes tool arguments as JSON
        strings. We parse them here before executing the search.
        """
        # Parse & validate parameters
        filter_dict = self._parse_metadata_filter(metadata_filter)
        k_int = self._parse_k(k)

        # Execute search
        retriever = self._get_retriever(filter_dict, k_int)
        docs = retriever.invoke(query)

        # Convert Document objects to plain data for JSON‑serializable return
        return [
            {
                "page_content": doc.page_content,
                "metadata": doc.metadata,
            }
            for doc in docs
        ]

    # ------------------------------------------------------------------ #
    # StructuredTool factory
    # ------------------------------------------------------------------ #

    def get_tools(self):
        """Return the LangChain StructuredTool for this vector store."""
        return [
            StructuredTool(
                name=f"search_in_vectorstore-{self.store_id}",
                func=self.search,
                description=(
                    "Semantic search over the vector store with optional metadata filtering "
                    "and top‑k override. Parameters must be strings (JSON for filter)."
                ),
                args_schema=SearchModel,
            )
        ]
