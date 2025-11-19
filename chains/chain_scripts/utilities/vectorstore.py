"""
vectorstore_toolkit.py

Tooling per ricerche semantiche su un vector store Chroma, esposto come
StructuredTool LangChain.

Caratteristiche principali
--------------------------
- Tutti i parametri del tool sono stringhe (compatibile con i tool call LLM).
- `query` è testo libero che viene embeddato e cercato nel vector store.
- `metadata_filter` è una stringa JSON che viene passata come `filter` a Chroma.
- `k` è il numero massimo di documenti da restituire (stringa che rappresenta un int).

Linee guida per l’agente
------------------------
1. Inizia con una chiamata esplorativa:
   - Usa una query significativa.
   - Lascia `metadata_filter` a "{}" o a `null` per vedere quali metadati sono
     disponibili nei risultati (es. `filename`, `page_name`, `source_context`).
2. Dopo aver visto i metadati, costruisci filtri mirati:
   - Esempio semplice: `{"filename": "report_2024.pdf"}`
   - Esempio con range: `{"page_number": {"$gte": 2}}`
   - Esempio combinato: `{"$and": [{"filename": "report_2024.pdf"}, {"page_name": "Summary"}]}`
3. Usa i filtri con intelligenza:
   - Preferisci pochi campi ben scelti (es. `filename`, `page_name`, `source_context`).
   - Evita liste enormi (es. centinaia di valori in `$in` o `$or`) che rallentano Chroma.
4. `k` controlla quante evidenze vuoi:
   - Usa valori piccoli (3–5) per “sbirciare” rapidamente.
   - Usa valori più grandi (10–20) quando ti serve una copertura più ampia.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any, List

from pydantic import BaseModel, Field, ValidationError
from langchain_core.tools import StructuredTool

# Registry helpers (assumed to wrap a Chroma vector store)
from vector_stores.api import vector_stores, load_vector_store

###############################################################################
# Helper to obtain (and lazily load) the vector store                        #
###############################################################################


def get_vectorstore_component(store_id: str):
    """Return a Chroma-based vector store instance from the global registry, loading it on-demand."""
    if store_id not in vector_stores:
        # The config id convention is <store_id>_config (change if different)
        load_vector_store(config_id=f"{store_id}_config")
    return vector_stores[store_id]


###############################################################################
# Metadata keys allowed in tool outputs                                      #
###############################################################################

# Solo questi metadati verranno restituiti nell'output dello strumento,
# se presenti nel Document.metadata.
ALLOWED_METADATA_KEYS = {
    "doc_store_collection",
    "doc_store_id",
    "filename",
    "filetype",
    "page_name",
    "page_number",
    "source",
    "source_context",
}

###############################################################################
# Pydantic schema for the tool                                               #
###############################################################################


class SearchModel(BaseModel):
    """Schema per i parametri *solo stringa* del tool di ricerca su Chroma.

    Uso concettuale del tool
    ------------------------
    - `query` è la descrizione testuale di ciò che stai cercando
      (domanda, riassunto del bisogno informativo, breve frase).
    - `metadata_filter` è una stringa JSON che descrive i vincoli sui metadati
      dei Document di Chroma (campo `filter` nei metodi di ricerca).
    - `k` è il numero massimo di documenti da restituire, espresso come stringa
      che rappresenta un intero positivo.

    Come usare i filtri (campo `metadata_filter`)
    ---------------------------------------------
    - Il valore deve essere una stringa che rappresenta un oggetto JSON,
      NON un array e NON una stringa con apici singoli.
      Esempio valido: '{"filename": "report_2024.pdf"}'
    - Il dict viene passato direttamente a Chroma come `filter` (where-clause).
    - Puoi usare:
        • uguaglianza: {"filename": "report_2024.pdf"}
        • range:       {"page_number": {"$gte": 2, "$lte": 5}}
        • logica AND:  {"$and": [{"filename": "report_2024.pdf"},
                                 {"page_name": "Summary"}]}
        • logica OR:   {"$or": [{"page_name": "Intro"},
                                {"page_name": "Summary"}]}
    - Se passi un oggetto piatto con più chiavi (es. {"a":1,"b":2}),
      Chroma lo interpreta come AND implicito.

    Strategia consigliata
    ---------------------
    1. Prima chiamata: `metadata_filter = "{}"` o omesso, `k` piccolo (es. "3"),
       per ispezionare i metadati disponibili.
    2. Chiamate successive: costruisci un filtro semplice ma discriminante usando
       soprattutto:
         - `filename`
         - `page_name`
         - `page_number`
         - `source_context`
    3. Evita filtri eccessivamente complessi o con liste enormi: sono lenti e
       difficili da mantenere per l’agente.
    """

    query: str = Field(
        ...,
        title="Query",
        description=(
            "Testo libero da cercare nel vector store Chroma. "
            "Riassumi qui cosa ti serve (domanda, concetto, breve descrizione), "
            "non incollare interi documenti."
        ),
    )
    metadata_filter: Optional[str] = Field(
        default="{}",
        title="Metadata filter (JSON string for Chroma)",
        description=(
            "Stringa JSON che rappresenta un oggetto di filtro metadati per Chroma "
            "(argomento `filter`). Esempi: \n"
            "  • Nessun filtro: \"{}\" \n"
            "  • Per filename: '{\"filename\": \"report_2024.pdf\"}' \n"
            "  • Per pagina:   '{\"page_number\": {\"$gte\": 2}}' \n"
            "  • Combinato:    '{\"$and\": ["
            "{\"filename\": \"report_2024.pdf\"}, "
            "{\"page_name\": \"Summary\"}]}' \n\n"
            "Il valore DEVE essere un oggetto JSON (non array, non stringa scalare). "
            "Usa pochi campi ben scelti (es. filename, page_name, page_number) "
            "per filtrare con efficienza."
        ),
    )
    k: Optional[str] = Field(
        default="10",
        title="k (stringified integer)",
        description=(
            "Numero massimo di documenti da restituire (come stringa di intero positivo). \n"
            "Suggerimenti: \n"
            "  • \"3\"–\"5\" per una prima esplorazione rapida \n"
            "  • \"10\"–\"20\" se ti serve una copertura più ampia \n"
            "Valori più grandi aumentano il costo e la quantità di testo restituito."
        ),
    )

    class Config:
        # Più robusto: l'agente non può inventare campi addizionali.
        extra = "forbid"


###############################################################################
# Toolkit manager                                                            #
###############################################################################


class VectorStoreToolKitManager:
    """Bridge between a Chroma vector store and a LangChain StructuredTool.

    - Gestisce il caricamento lazy del vector store associato a `store_id`.
    - Converte parametri stringa (come passati dai tool calls LLM) in valori
      tipizzati per la ricerca (dict per i filtri, int per `k`).
    - Restituisce una lista di documenti in formato JSON-serializzabile con:
        • `page_content`: testo del Document
        • `metadata`: solo un sottoinsieme controllato di metadati utili
          (vedi `ALLOWED_METADATA_KEYS`).
    """

    def __init__(
        self,
        store_id: str,
        search_type: str = "similarity",
        default_k: int = 10,
        search_kwargs: Optional[Dict[str, Any]] = None,
    ) -> None:
        """
        Parameters
        ----------
        store_id:
            Identificativo logico del vector store Chroma, usato per recuperare
            l'istanza dal registry globale.
        search_type:
            Tipo di ricerca del retriever (es. 'similarity', 'mmr', ...).
        default_k:
            Valore di fallback per `k` se non specificato a livello di tool call.
        search_kwargs:
            Argomenti base passati al retriever; possono essere sovrascritti
            per singola chiamata (es. con filtri e k diversi).
        """
        self.store_id = store_id
        self.vectorstore = get_vectorstore_component(store_id=store_id)
        self.search_type = search_type
        # baseline kwargs used for *every* call; may be overridden per-call
        self.base_search_kwargs: Dict[str, Any] = search_kwargs or {}
        self.base_search_kwargs.setdefault("k", default_k)

    # ------------------------------------------------------------------ #
    # Internal helpers                                                   #
    # ------------------------------------------------------------------ #

    @staticmethod
    def _parse_metadata_filter(filter_str: Optional[str]) -> Optional[Dict[str, Any]]:
        """Convert a JSON string into a dict suitable as Chroma `filter`.

        - Se `filter_str` è vuota o None → ritorna None (nessun filtro).
        - Se `filter_str` è un oggetto JSON vuoto (`{}`) → ritorna None
          così il campo `filter` NON viene passato a Chroma.
        - Se il parsing fallisce o non è un oggetto JSON → solleva ValueError
          con un messaggio comprensibile all'agente.
        """
        if not filter_str:
            return None
        try:
            parsed = json.loads(filter_str)
            if not isinstance(parsed, dict):
                raise ValueError(
                    "metadata_filter must decode to a JSON object/dict, "
                    "as required by Chroma `filter`."
                )
            # 🔹 NOVITÀ: se il dict è vuoto ({}), lo trattiamo come "nessun filtro"
            if not parsed:  # equivalente a: if parsed == {}
                return None
            return parsed
        except (json.JSONDecodeError, ValueError) as exc:
            raise ValueError(
                f"Invalid 'metadata_filter' JSON: {exc}. "
                "Expected a JSON object like '{\"year\": 2024}' or "
                "'{\"filename\": \"report.pdf\"}'."
            ) from exc


    @staticmethod
    def _parse_k(k_str: Optional[str]) -> Optional[int]:
        """Convert `k` from string to positive int (or None if not provided)."""
        if k_str in (None, ""):
            return None
        try:
            k_int = int(k_str)
            if k_int <= 0:
                raise ValueError("k must be positive")
            return k_int
        except ValueError as exc:
            raise ValueError(
                f"Invalid 'k': {exc}. Provide a positive integer as string "
                "(e.g. \"3\", \"10\")."
            ) from exc

    def _build_search_kwargs(
        self,
        metadata_filter: Optional[Dict[str, Any]],
        k: Optional[int],
    ) -> Dict[str, Any]:
        """Merge base kwargs with per-call overrides (filter, k) for Chroma retriever."""
        kwargs = dict(self.base_search_kwargs)  # shallow copy
        if metadata_filter is not None:
            kwargs["filter"] = metadata_filter
        if k is not None:
            kwargs["k"] = k
        return kwargs

    # ------------------------------------------------------------------ #
    # Core retrieval logic                                               #
    # ------------------------------------------------------------------ #

    def _get_retriever(
        self,
        metadata_filter: Optional[Dict[str, Any]] = None,
        k: Optional[int] = None,
    ):
        """Costruisce un retriever LangChain sul vector store Chroma."""
        search_kwargs = self._build_search_kwargs(metadata_filter, k)
        return self.vectorstore.as_retriever(
            search_type=self.search_type,
            search_kwargs=search_kwargs,
        )

    # ------------------------------------------------------------------ #
    # Public method to expose via the tool                               #
    # ------------------------------------------------------------------ #

    def search(
        self,
        query: str,
        metadata_filter: Optional[str] = None,
        k: Optional[str] = None,
    ) -> Any:
        """Esegue una ricerca semantica su Chroma e restituisce i documenti trovati.

        Parametri (tutti stringhe, come arrivano dal tool call):
        - query: testo libero da embeddare e cercare nel vector store.
        - metadata_filter: stringa JSON con i vincoli sui metadati (Chroma `filter`).
        - k: massimo numero di documenti da restituire (stringa che rappresenta un int).

        Output:
        - Lista di dict, ciascuno con:
            • `page_content`: il testo del Document.
            • `metadata`: SOLO i metadati in ALLOWED_METADATA_KEYS, se presenti.
        """

        try:
            # Parse & validate parameters
            filter_dict = self._parse_metadata_filter(metadata_filter)
            k_int = self._parse_k(k)

            # Execute search on Chroma
            retriever = self._get_retriever(metadata_filter=filter_dict, k=k_int)
            docs = retriever.invoke(query)

            results: List[Dict[str, Any]] = []

            for doc in docs:
                # Copia dei metadata per non modificare l'oggetto originale
                original_metadata = dict(doc.metadata or {})

                # Rimuoviamo esplicitamente 'orig_elements' se presente
                original_metadata.pop("orig_elements", None)

                # Filtriamo i metadati: solo quelli esplicitamente ammessi
                filtered_metadata = {
                    key: value
                    for key, value in original_metadata.items()
                    if key in ALLOWED_METADATA_KEYS
                }

                results.append(
                    {
                        "page_content": doc.page_content,
                        "metadata": filtered_metadata,
                    }
                )

        except Exception as e:
            # In caso di errori, ritorniamo una stringa (comportamento originale
            # utile per l'agente: può leggere e adattare la chiamata successiva).
            return f"Error: {e}"

        return results

    # ------------------------------------------------------------------ #
    # StructuredTool factory                                             #
    # ------------------------------------------------------------------ #

    def get_tools(self):
        """Crea la lista di StructuredTool da registrare in LangChain."""

        return [
            StructuredTool(
                name=f"search_in_vectorstore-{self.store_id}",
                func=self.search,
                description=(
                    f"Semantic vector search over the Chroma collection '{self.store_id}'.\n\n"
                    "Intended usage:\n"
                    "  • Provide a concise natural language `query` describing what you need.\n"
                    "  • Optionally provide a JSON `metadata_filter` string to constrain the search "
                    "    by fields such as `filename`, `page_name`, `page_number`, `source_context`.\n"
                    "  • `k` controls how many top documents are returned (as a stringified integer).\n\n"
                    "Good patterns:\n"
                    "  1) First call with an empty filter (\"{}\") and small k (e.g. \"3\") to inspect "
                    "     the available metadata.\n"
                    "  2) Subsequent calls with a simple, discriminative filter, for example:\n"
                    "       '{\"filename\": \"report_2024.pdf\"}' or\n"
                    "       '{\"$and\":[{\"filename\":\"report_2024.pdf\"},"
                    "{\"page_name\":\"Summary\"}]}'\n"
                    "  3) Avoid overly complex filters with huge `$in` or `$or` lists: they are slow "
                    "     and rarely necessary.\n"
                ),
                args_schema=SearchModel,
            )
        ]
