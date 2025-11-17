
########################################################################################################################

import json
import os

from fastapi import FastAPI, HTTPException, Path, Body, APIRouter
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.messages import HumanMessage, BaseMessage
from pydantic import BaseModel, Field
from typing import Dict, Any, List, Optional
from pymongo import MongoClient
from chains.utilities.chain_manager import ChainManager
from fastapi.responses import StreamingResponse

#from langchain_community.callbacks.manager import get_openai_callback
from langchain_community.callbacks import get_openai_callback

from chains.utilities.multimodal import to_message, build_parts, build_parts_legacy

router = APIRouter()
MONGO_CONNECTION_STRING = os.getenv('MONGO_CONNECTION_STRING', 'localhost')
client = MongoClient(MONGO_CONNECTION_STRING)
db = client['chain_db']
collection = db['chain_configs']
chain_manager = ChainManager(collection)


class ChainConfigRequest(BaseModel):
    chain_type: str = Field("agent_with_tools", example="qa_chain", title="Chain Type", description="The chain's type to configure.")
    config_id: str = Field(..., example="example_chain_config", title="Config ID", description="The unique ID of the chain configuration.")
    chain_id: str = Field(..., example="example_chain", title="Chain ID", description="The unique ID of the chain.")
    prompt_id: str = Field("prompt_id", example="example_prompt", title="Prompt ID", description="The unique ID of the prompt.")
    system_message: str = Field("you are an helpful assistant", example="example_system_message", title="System Message", description="The system message of the agent.")
    llm_id: str = Field("llm_id", example="llm_id", title="LLM ID", description="The unique ID of the LLM.")
    vectorstore_id: str = Field("vectorstore_id", example="example_vectorstore", title="Vectorstore ID", description="The unique ID of the vectorstore.")
    tools: List[Dict[str, Any]] = Field([], example="example_tools", title="Vectorstore ID", description="The unique ID of the vectorstore.")
    extra_metadata : Optional[Dict[str, Any]] = Field(
        default=None,
        description="Arbitrary key-value pairs with additional metadata."
    )

#class ExecuteChainRequest(BaseModel):
#    chain_id: str = Field(..., example="example_chain", title="Chain ID", description="The unique ID of the chain to execute.")
#    query: Dict[str, Any] = Field(..., example={"input": "What is my name?", "chat_history": [["user", "hello, my name is mario!"], ["assistant", "hello, how are you mario?"]]}, title="Query", description="The input query for the chain.")
#    inference_kwargs: Dict[str, Any] = Field(..., example={}, description="")


class ExecuteChainRequest(BaseModel):
    """
    Modello di richiesta per l'esecuzione di una chain.

    Questo modello supporta due modalità di input:
    1. **Legacy** tramite il campo `query` (deprecato): un dizionario con chiavi
       - `input`: stringa di testo
       - `chat_history`: lista di tuple [ruolo, messaggio]
    2. **Multimodale** tramite i nuovi campi:
       - `input_text`: testo dell'ultimo turno
       - `input_images`: lista di URL o data-URI per immagini nell'ultimo turno
       - `chat_history`: lista di messaggi storici, ciascuno con:
         - `role`: "user" o "assistant"
         - `parts`: array di parti multimodali (dict con `type`, `text` o `image_url`)

    **Deprecazione:** il campo `query` sarà rimosso in una futura versione. Si consiglia di migrare all'input multimodale.
    """
    chain_id: str = Field(
        ...,
        example="example_chain",
        title="Chain ID",
        description="ID univoco della chain da eseguire."
    )

    # ---------------- Legacy query (deprecata) ----------------
    query: Optional[Dict[str, Any]] = Field(
        None,
        example={
            "input": "What is my name?",
            "chat_history": [
                ["user", "hello, my name is mario!"],
                ["assistant", "hello, how are you mario?"]
            ]
        },
        title="Legacy Query",
        description=(
            "(DEPRECATO) Dizionario di input testuale e cronologia chat. "
            "Usare `input_text`, `input_images` e `chat_history` invece."
        )
    )

    # ------------- Nuovi campi multimodali -------------------
    input_text: Optional[str] = Field(
        None,
        example="What is my dog breed?",
        title="Input Text",
        description="Testo del messaggio corrente dell'utente."
    )
    input_images: Optional[List[Dict[str, Any]]] = Field(
        None,
        example=["https://example.com/dog.jpg"],
        title="Input Images",
        description=(
            "Lista di URL o data-URI per immagini da includere "
            "nel messaggio corrente."
        )
    )
    chat_history: Optional[List[Dict[str, Any]]] = Field(
        None,
        example=[
            {
                "role": "user",
                "parts": [
                    {"type": "text", "text": "Here is my dog"},
                    {"type": "image_url", "image_url": {"url": "https://.../dog.jpg"}}
                ]
            },
            {
                "role": "assistant",
                "parts": [
                    {"type": "text", "text": "Looks like a Labrador!"}
                ]
            }
        ],
        title="Chat History",
        description=(
            "Cronologia dei messaggi precedenti con parti multimodali. "
            "Ogni messaggio è un dict con chiavi 'role' e 'parts'."
        )
    )

    inference_kwargs: Dict[str, Any] = Field(
        default_factory=dict,
        example={"temperature": 0.2, "stream": True},
        description="Argomenti addizionali per l'invocazione della chain."
    )


@router.post("/configure_chain/", response_model=dict)
async def configure_chain(request: ChainConfigRequest):
    """
    Configure a new chain and store its configuration.

    This endpoint allows you to configure a new chain by specifying the configuration ID, chain ID, prompt ID, LLM ID, and vectorstore ID.
    The configuration is then stored in MongoDB.

    - **request**: A JSON object containing the configuration details.

    Returns:
    - **config_id**: The ID of the newly created configuration.
    """
    try:
        result = chain_manager.configure_chain(request.dict())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.put("/update_chain_config/{config_id}", response_model=dict)
async def update_chain_config(config_id: str, request: ChainConfigRequest):
    """
    Update an existing chain configuration.

    This endpoint updates the configuration of an existing chain by specifying the configuration ID, chain ID, prompt ID, LLM ID, and vectorstore ID.

    - **config_id**: The ID of the configuration to update.
    - **request**: A JSON object containing the updated configuration details.

    Returns:
    - **config_id**: The ID of the updated configuration.
    """
    try:
        result = chain_manager.update_chain_config(config_id, request.dict())
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.delete("/delete_chain_config/{config_id}", response_model=dict)
async def delete_chain_config(config_id: str = Path(..., description="The unique ID of the chain configuration to delete.")):
    """
    Delete a specific chain configuration.

    This endpoint deletes the configuration details of a specific chain using the configuration ID.

    - **config_id**: The unique ID of the chain configuration to delete.

    Returns:
    - A confirmation message upon successful deletion.
    """
    try:
        result = chain_manager.delete_chain_config(config_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/load_chain/{config_id}", response_model=dict)
async def load_chain(config_id: str = Path(..., description="The unique ID of the chain configuration to load.")):
    """
    Load a chain into memory based on its configuration ID.

    This endpoint loads a chain into memory using the specified configuration ID. If the chain is already loaded, an error is returned.

    - **config_id**: The unique ID of the chain configuration to load.

    Returns:
    - A confirmation message upon successful loading.
    """
    try:
        result = chain_manager.load_chain(config_id)
        return result
    except ValueError as e:
        print(e)
        raise HTTPException(status_code=400, detail=str(e))

@router.post("/unload_chain/{chain_id}", response_model=dict)
async def unload_chain(chain_id: str = Path(..., description="The unique ID of the chain to unload.")):
    """
    Unload a chain from memory.

    This endpoint unloads a chain from memory using the specified chain ID. If the chain is not found, an error is returned.

    - **chain_id**: The unique ID of the chain to unload.

    Returns:
    - A confirmation message upon successful unloading.
    """
    try:
        result = chain_manager.unload_chain(chain_id)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

@router.get("/list_loaded_chains/", response_model=List[str])
async def list_loaded_chains():
    """
    List all currently loaded chains.

    This endpoint retrieves and returns a list of all currently loaded chains.

    Returns:
    - A list of chain IDs for the currently loaded chains.
    """
    return chain_manager.list_loaded_chains()

@router.get("/list_chain_configs/", response_model=List[Dict[str, Any]])
async def list_chain_configs():
    """
    List all chain configurations.

    This endpoint retrieves and returns a list of all chain configurations stored in MongoDB.

    Returns:
    - A list of chain configurations.
    """
    try:
        configs = chain_manager.list_chain_configs()
        return configs
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/chain_config/{config_id}", response_model=Dict[str, Any])
async def get_chain_config(config_id: str = Path(..., description="The unique ID of the chain configuration to retrieve.")):
    """
    Retrieve a specific chain configuration.

    This endpoint retrieves the configuration details of a specific chain using the configuration ID.

    - **config_id**: The unique ID of the chain configuration to retrieve.

    Returns:
    - The configuration details of the specified chain.
    """
    try:
        config = chain_manager.get_chain_config(config_id)
        return config
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/execute_chain/", response_model=dict)
async def execute_chain(request: ExecuteChainRequest):

    """
    Execute a specific chain.

    This endpoint executes a loaded chain using the provided query.

    - **request**: A JSON object containing the chain ID and query.

    Returns:
    - The result of the chain execution.
    """

    # TODO:
    #  - integra tracciamento di token e costi
    #  - integra caricamento automatico dell oggetto se non presente in memoria (default true, da settare mediante input)

    try:
        chain = chain_manager.get_chain(request.chain_id)
        with get_openai_callback() as cb:
            result = chain.invoke(request.query, **request.inference_kwargs)
            print(result)
            print("\n\nToken usage:\n")
            print(cb)
        return result
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stream_chain")
async def stream_chain(request: ExecuteChainRequest):

    # TODO:
    #  - integra tracciamento di token e costi
    #  - integra caricamento automatico dell oggetto se non presente in memoria (default true, da settare mediante input)

    async def generate_response(chain: Any, query: Dict[str, Any], inference_kwargs: Dict[str, Any], stream_only_content: bool = False):
        with get_openai_callback() as cb:
            async for chunk in chain.astream(query, **inference_kwargs):

                try:
                    print("#"*120)
                    print(chunk)
                    print("#" * 120)
                    chunk = json.dumps(chunk, indent=2)
                except Exception as e:
                    #print(e)
                    print(chunk)
                    chunk = {"error": "output object not serializable"}
                    chunk = json.dumps(chunk, indent=2)
                yield chunk
                print("\n\nToken usage:\n")
                print(cb)
                # yield cb

    try:
        #body = request

        #if not chain_manager.chains.get(body.chain_id):
        #    await load_chain(config_id=f"{body.chain_id}_config")

        #chain = chain_manager.get_chain(body.chain_id)
        #query = body.query
        #inference_kwargs = body.inference_kwargs
        # ✅ lazy‑load automatico: pensa a tutto ChainManager.get_chain
        chain = chain_manager.get_chain(request.chain_id)
        query = request.query
        inference_kwargs = request.inference_kwargs

        return StreamingResponse(generate_response(chain, query, inference_kwargs), media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

# ─────────────────────────────────────────────────────────────────────────────
# Helper: rimuove i payload pesanti dagli eventi prima di serializzarli
# ─────────────────────────────────────────────────────────────────────────────
def _sanitize_event(evt: Dict[str, Any]) -> Dict[str, Any]:
    """
    Restituisce una copia dell'evento `evt` in cui, se presente,
    `evt["data"]["output"]` viene sostituito con la stringa fissa
    "Contenuto non mostrato..".

    Puoi estendere facilmente la funzione per filtrare anche altri campi
    (es. `input` in on_tool_start) senza rompere nulla.
    """
    safe_evt = dict(evt)                       # shallow-copy
    data = dict(safe_evt.get("data", {}))      # copia del sotto-dict
    #if "output" in data:                       # se presente ➜ sostituisci
        #data["output"] = "Contenuto non mostrato.."
    safe_evt["data"] = data
    return safe_evt

#@router.post("/stream_events_chain")
async def stream_events_chain(request: ExecuteChainRequest):

    # TODO:
    #  - integra tracciamento di token e costi
    #  - integra caricamento automatico dell oggetto se non presente in memoria (default true, da settare mediante input)

    async def generate_response(chain: Any, query: Dict[str, Any], inference_kwargs: Dict[str, Any], stream_only_content: bool = False):

        with get_openai_callback() as cb:
            async for event in chain.astream_events(
                query,
                version="v1",
                **inference_kwargs,
            ):
                kind = event["event"]
                if kind == "on_chain_start":
                    if (
                            event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        print(
                            f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                        )
                elif kind == "on_chain_end":
                    if (
                            event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        print()
                        print("--")
                        print(
                            f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                        )
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Empty content in the context of OpenAI means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        print(content, end="|")
                        yield content
                    else:
                        yield ""

                elif kind == "on_tool_start":
                    print("--")
                    print(
                        f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                    )
                    yield json.dumps(event)

                elif kind == "on_tool_end":
                    print(f"Done tool: {event['name']}")
                    print(f"Tool output was: {event['data'].get('output')}")
                    print("--")
                    yield json.dumps(_sanitize_event(event))

            print("\n\nToken usage:\n")
            print(str(cb))
            #yield cb

    try:
        #body = request

        #if not chain_manager.chains.get(body.chain_id):
        #    await load_chain(config_id=f"{body.chain_id}_config")

        #chain = chain_manager.get_chain(body.chain_id)
        #query = body.query
        #inference_kwargs = body.inference_kwargs

        chain = chain_manager.get_chain(request.chain_id)
        query = request.query
        inference_kwargs = request.inference_kwargs

        return StreamingResponse(generate_response(chain, query, inference_kwargs), media_type="application/json")
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/stream_events_chain")
async def stream_events_chain(request: ExecuteChainRequest):

    # TODO:
    #  - integra tracciamento di token e costi
    #  - integra caricamento automatico dell oggetto se non presente in memoria (default true, da settare mediante input)

    async def generate_response(chain: Any, query: Dict[str, Any], inference_kwargs: Dict[str, Any], stream_only_content: bool = False):

        with get_openai_callback() as cb:
            async for event in chain.astream_events(
                query,
                version="v1",
                **inference_kwargs,
            ):
                kind = event["event"]
                if kind == "on_chain_start":
                    if (
                            event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        print(
                            f"Starting agent: {event['name']} with input: {event['data'].get('input')}"
                        )
                elif kind == "on_chain_end":
                    if (
                            event["name"] == "Agent"
                    ):  # Was assigned when creating the agent with `.with_config({"run_name": "Agent"})`
                        print()
                        print("--")
                        print(
                            f"Done agent: {event['name']} with output: {event['data'].get('output')['output']}"
                        )
                if kind == "on_chat_model_stream":
                    content = event["data"]["chunk"].content
                    if content:
                        # Empty content in the context of OpenAI means
                        # that the model is asking for a tool to be invoked.
                        # So we only print non-empty content
                        print(content, end="|")
                        yield content
                    else:
                        yield ""

                elif kind == "on_tool_start":
                    print("--")
                    print(
                        f"Starting tool: {event['name']} with inputs: {event['data'].get('input')}"
                    )
                    yield json.dumps(event)

                elif kind == "on_tool_end":
                    print(f"Done tool: {event['name']}")
                    print(f"Tool output was: {event['data'].get('output')}")
                    print("--")
                    yield json.dumps(_sanitize_event(event))

                else:
                    event_name = event.get('name',None)
                    print(f"Untracked event: {event_name}")
                    print(event)
                    print("--")


            print("\n\nToken usage:\n")
            print(str(cb))
            #yield cb

    try:
        chain = chain_manager.get_chain(request.chain_id)
        inference_kwargs = request.inference_kwargs

        # —————— fallback legacy vs multimodale ——————
        if request.query is not None:
            q = request.query
            # 1) Legacy “stringa + lista di dict” o “stringa + lista di liste”
            if isinstance(q.get("input"), str):
                # costruisco il primo HumanMessage
                user_parts = build_parts_legacy(q["input"], [])
                user_msg = HumanMessage(content=user_parts)

                history_msgs: List[BaseMessage] = []
                for item in q.get("chat_history", []):
                    # Se è una tupla/lista [role, text]
                    if isinstance(item, (list, tuple)) and len(item) == 2:
                        role, txt = item
                    # Se è un dict {id, role, content, …}
                    elif isinstance(item, dict) and "role" in item and "content" in item:
                        role, txt = item["role"], item["content"]
                    else:
                        # Skip formati sconosciuti
                        continue
                    obj = {"role": role, "parts": [{"type": "text", "text": txt}]}
                    history_msgs.append(to_message(obj))

                model_query = {"input": [user_msg], "chat_history": history_msgs}

            # 2) Legacy già “nuovo”: input è lista di BaseMessage, history lista di BaseMessage/dict
            #elif isinstance(q.get("input"), list):
            #    model_query = q
            else:
                # 1b) già in formato dict con input:list e chat_history:list → usalo così com'è
                model_query = q
        else:
            # 2) nuovo path multimodale
            user_msg: BaseMessage = HumanMessage(
                content=build_parts(request.input_text, request.input_images)
            )
            history_msgs: List[BaseMessage] = [
                to_message(m) for m in (request.chat_history or [])
            ]
            model_query = {
                "input": [user_msg],
                "chat_history": history_msgs,
            }

        return StreamingResponse(
            generate_response(chain, model_query, inference_kwargs),
            media_type="application/json"
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    import uvicorn

    app = FastAPI()

    app.include_router(router, prefix="/chains", tags=["chains"])

    uvicorn.run(app, host="127.0.0.1", port=8100)

