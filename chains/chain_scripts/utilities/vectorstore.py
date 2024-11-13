import json
import os
from pymongo import MongoClient
from pydantic import BaseModel, Field
from typing import Optional, Any, Dict
from langchain_core.tools import StructuredTool
from vector_stores.api import vector_stores, load_vector_store


def get_vectorstore_component(store_id: str):
    # Function to get a vectorstore component by ID

    if not vector_stores.get(store_id):
        try:
            load_vector_store(config_id=f"{store_id}_config")
        except Exception as e:
            print(e)

    vector_store = vector_stores[store_id]

    return vector_store


class SearchModel(BaseModel):
    query: str = Field("input query", title="Query", description="Query di input impiegata per effettuare la ricerca nel vector store.")


class VectorStoreToolKitManager:
    def __init__(self,
                 store_id: str,
                 search_type: str = "similarity",
                 search_kwargs: Optional[Dict[str, Any]] = {"k": 10}):
        
        """Inizializza VectorStoreToolKit."""
        self.store_id = store_id
        self.vectorstore = get_vectorstore_component(store_id=store_id)
        self.search_type = search_type
        self.search_kwargs = search_kwargs

    def as_retriever(self):

        retriever = self.vectorstore.as_retriever(**{
            "search_type": self.search_type,
            "search_kwargs": self.search_kwargs
        })

        return retriever

    def search(self, query: str):
        return str(self.as_retriever().invoke(input=query))

    def get_tools(self):
        """Restituisce una lista degli strumenti configurati usando StructuredTool."""
        return [
            StructuredTool(
                name=f"search_in_vectorstore-{self.store_id}",
                func=self.search,
                description=f"Use this tool to search relevant docs in vector store with id {self.store_id}.",
                args_schema=SearchModel
            )
        ]


# Esempio di utilizzo
#if __name__ == "__main__":
    # Inizializza MongoDBToolKit con una connection string
#    mongo_toolkit = MongoDBToolKitManager(connection_string="mongodb://localhost:27017", default_database="sans7-database_0")

    # Definisci i dati da inserire
#    write_data = WriteDataModel(collection_name="example_collection", data={"name": "Alice", "age": 30})

    # Inserisci il documento nella collection
#    result = mongo_toolkit.write_to_mongo(write_data)
#    print(result)

    # Leggi il documento inserito
#    read_data = ReadDataModel(collection_name="example_collection", query={"name": "Alice"})
#    documents = mongo_toolkit.read_from_mongo(read_data)
#    print(documents)
