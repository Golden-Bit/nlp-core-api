"""
agent_gpt4o_tools.py

Script “chiavi in mano” per un agente LangChain con GPT‑4o, strumenti HTTP
di default e possibilità di estendere lo stack con altri StructuredTool.
Contiene anche un esempio di utilizzo con streaming eventi.

Requisiti:
    pip install langchain langchain-openai langchain-community
"""
import base64
import json
import os
import asyncio
from typing import List, Optional, Any, Dict
import nest_asyncio
import requests
from langchain_community.utilities.requests import RequestsWrapper
from langchain_core.tools import BaseTool
from langchain_openai import ChatOpenAI
from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder, HumanMessagePromptTemplate
from langchain.tools import StructuredTool
from langchain_community.tools import (
    RequestsGetTool,
    RequestsPostTool,
    RequestsPatchTool,   # nuovo
    RequestsPutTool,     # nuovo
    RequestsDeleteTool   # nuovo
)
from langchain_community.utilities.requests import RequestsWrapper
from pydantic import BaseModel, Field


class GenerateResponseInput(BaseModel):
    """Modello di input per il tool generate_response."""
    input: str = Field(...,description="input message to send")


# Applica patch per permettere asyncio.run dentro un loop già attivo
nest_asyncio.apply()

def _default_http_tools(requests_wrapper: RequestsWrapper):
    return [
        RequestsGetTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
        RequestsPostTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
        RequestsPutTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
        RequestsPatchTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
        RequestsDeleteTool(requests_wrapper=requests_wrapper, allow_dangerous_requests=True),
    ]

# --------------------------------------------------------------------------- #
# 1. Configurazione variabile d’ambiente                                       #
# --------------------------------------------------------------------------- #


# --------------------------------------------------------------------------- #
# 2. System‑message (modificabile)                                             #
# --------------------------------------------------------------------------- #

DETAILED_SYSTEM_MESSAGE = """
You are an advanced GPT‑4o assistant embedded in a LangChain agent.

Your mission:
• Provide helpful, precise and concise answers.
• Decide autonomously whether calling a tool is required:
  – If the user requests data obtainable via an available tool, call it.
  – If no tool is appropriate, answer directly.
• After using a tool, ALWAYS incorporate its result in natural‑language form
  before ending the reply.
• Use professional tone, markdown formatting, short paragraphs, avoid fluff.
• Do not expose internal chain‑of‑thought or code unless explicitly asked.

#---INSTRUCTIONS---#
ALWAYS FOLLOW THE FOLLOWING STEPS:

    1. ANALYZE THE USER’S REQUEST AND UNDERSTAND HOW TO PROCEED TO ACHIEVE THE OBJECTIVE, UTILIZING THE PROVIDED TOOLS TO SEND REQUESTS TO THE API.

    2. ALWAYS MAKE A CALL TO LIST ENDPOINTS TO OBTAIN A LIST OF PATHS; FOR EACH PATH, WE OBTAIN A SUMMARY FOR THE VARIOUS METHODS AVAILABLE (GET, POST, ETC.).

    3. BASED ON THE DESCRIPTIONS AND CONTENT OBTAINED FROM THE LIST OF ENDPOINTS, DETERMINE THE CORRECT PATH AND METHOD(S) TO USE TO EXECUTE THE REQUEST. NEVER EXECUTE A PATH/METHOD COMBINATION THAT IS NOT INCLUDED IN THE ‘LIST ENDPOINTS’ RESULTS.

    4. THEN PASS THE PATH TO THE TOOL USED TO RETRIEVE DETAILED DESCRIPTIONS OF THE ENDPOINT FOR THE PROVIDED PATH.

    5. BASED ON THE PREVIOUS RESULTS AND THE DATA PROVIDED BY THE USER, EXECUTE THE API REQUEST, PROVIDING THE DATA IN A MANNER CONSISTENT WITH THE SCHEMA.

    6. IF ADDITIONAL API CALLS ARE REQUIRED, REPEAT THE CYCLE.
"""


# --------------------------------------------------------------------------- #
# 3. Funzione di factory per l’agente                                          #
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Helpers per OpenAPI                                                         #
# --------------------------------------------------------------------------- #
import copy, pathlib, json
from typing import Union, Dict, Any
import yaml, jsonref

def load_spec(src: Union[str, pathlib.Path, dict]) -> Dict[str, Any]:
    """
    Carica lo spec OpenAPI da:
      • dict già in memoria
      • percorso file locale (.json/.yaml/.yml)
      • URL HTTP(S) a file .json/.yaml/.yml
      • stringa Base64 che codifica JSON o YAML
    Restituisce sempre un dict Python.
    """
    # 1️⃣ Se è già un dict, ne restituisco una copia
    if isinstance(src, dict):
        return copy.deepcopy(src)

    s = str(src).strip()

    # 2️⃣ Se è un URL HTTP(S), lo scarico
    if s.startswith("http://") or s.startswith("https://"):
        resp = requests.get(s)
        resp.raise_for_status()
        text = resp.text

    # 3️⃣ Se sembra Base64 (charset e lunghezza multipla di 4), cerco di decodificarlo
    elif all(c in "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/=\n\r" for c in s) and len(s) % 4 == 0:
        try:
            decoded = base64.b64decode(s)
            text = decoded.decode("utf-8")
        except Exception:
            # fallback: trattalo come stringa raw
            text = s

    # 4️⃣ Altrimenti, se è un path esistente, leggo il file, altrimenti lo tratto come testo raw
    else:
        path = pathlib.Path(s)
        if path.exists():
            text = path.read_text("utf-8")
        else:
            text = s

    # Provo prima JSON, altrimenti YAML
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return yaml.safe_load(text)


def dereference_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Risolve i riferimenti $ref (anche profondi) restituendo
    una copia completamente “flattened”.
    """
    return jsonref.replace_refs(spec)

# --------------------------------------------------------------------------- #
# Generazione dinamica di StructuredTool da spec                             #
# --------------------------------------------------------------------------- #
from langchain.tools import StructuredTool
# --------------------------------------------------------------------------- #
# 0. Costanti/metodi HTTP e helper riutilizzabili                              #
# --------------------------------------------------------------------------- #
HTTP_METHODS = {"get", "post", "put", "patch", "delete", "options", "head", "trace"}

def method_summaries(path_item: Dict[str, Any]) -> List[str]:
    """
    Estrae le summary/description presenti all’interno dei metodi HTTP
    di un singolo path OpenAPI.
    Restituisce una lista di stringhe formattate es. ["GET: Elenca…", "POST: Crea…"].
    """
    summaries = []
    for m, m_def in path_item.items():
        if m.lower() in HTTP_METHODS:
            txt = m_def.get("summary") or m_def.get("description") or ""
            if txt:
                summaries.append(f"{m.upper()}: {txt.strip()}")
    return summaries

def make_list_endpoints_tool(flat_spec: Dict[str, Any]) -> StructuredTool:
    endpoints = []
    for path, item in flat_spec.get("paths", {}).items():
        summaries = method_summaries(item)
        endpoints.append(
            {
                "path": path,
                # concateno più summary (GET|POST …) con “ | ”, altrimenti stringa vuota
                "summary": " | ".join(summaries) if summaries else ""
            }
        )

    def _list(_: str = "") -> Dict[str, Any]:
        return {"endpoints": endpoints}

    return StructuredTool.from_function(
        name="list_endpoints",
        func=_list,
        description="Elenca tutti gli endpoint con path e summary (metodo‑livello)."
    )


'''def make_endpoint_search_tool(flat_spec: Dict[str, Any]) -> StructuredTool:
    cache = {
        p: (info.get("summary") or info.get("description") or "")
        for p, info in flat_spec.get("paths", {}).items()
    }

    def _search(query: str = "") -> Dict[str, Any]:
        if not query:
            return make_list_endpoints_tool(flat_spec).func()
        q = query.lower()
        matches = [
            {"path": p, "summary": s}
            for p, s in cache.items()
            if q in p.lower() or q in s.lower()
        ]
        return {"matches": matches}

    return StructuredTool.from_function(
        name="endpoint_search",
        func=_search,
        description="Cerca endpoint il cui path o summary contiene la query."
    )'''

def make_endpoint_details_tool(flat_spec: Dict[str, Any]) -> StructuredTool:
    paths = flat_spec.get("paths", {})

    def _details(path_query: str) -> Dict[str, Any]:
        if path_query in paths:
            return {"exact": path_query, "definition": paths[path_query]}
        candidates = [p for p in paths if path_query in p]
        if candidates:
            return {
                "error": "ambiguous_path",
                "candidates": candidates,
                "message": f"Il path '{path_query}' non è univoco."
            }
        return {"error": "not_found", "message": f"Nessun endpoint contiene '{path_query}'."}

    return StructuredTool.from_function(
        name="endpoint_details",
        func=_details,
        description="Ritorna la definizione OpenAPI completa di un path."
    )

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
    if "output" in data:                       # se presente ➜ sostituisci
        data["output"] = "Contenuto non mostrato.."
    safe_evt["data"] = data
    return safe_evt

# --------------------------------------------------------------------------- #
# 3. Funzione di factory per l’agente                                          #
# --------------------------------------------------------------------------- #
def build_gpt4o_agent(
    *,
    extra_tools: Optional[List[StructuredTool]] = None,
    openapi_spec: Optional[Union[str, pathlib.Path, dict]] = None,
    server_url: Optional[str] = None,
    system_message: str = DETAILED_SYSTEM_MESSAGE,
    model_name: str = "gpt-4o-mini",
    openai_api_key="",
    model_kwargs: Optional[Dict[str, Any]] = None,
    verbose: bool = True,
) -> AgentExecutor | Any:
    """
    Args aggiunti:
        openapi_spec : percorso YAML/JSON, stringa o dict con lo spec.
        server_url   : base‑url da usare per tutte le richieste HTTP
                       (es. 'https://api.acme.com/v1').
    """
    dynamic_footer = ""  # verrà concatenato in fondo al system_message


    # 0.1 – URL del server
    if server_url:
        dynamic_footer += (
            "\n\n---\n"
            "**SERVER URL PER LE CHIAMATE HTTP**\n"
            f"→ `{server_url}`\n"
        )

    # 0.2 – Descrizioni endpoint (se è stato passato uno spec)
    flat_spec = None  # lo useremo anche più avanti
    if openapi_spec:
        raw_spec = load_spec(openapi_spec)
        flat_spec = dereference_spec(raw_spec)

        api_description = (raw_spec.get("info", {}).get("description") or "").strip()
        if api_description:
            dynamic_footer += (
                "\n\n---\n"
                "**DESCRIZIONE API (dallo spec)**\n"
                f"{api_description}\n"
            )

        #descrizioni = [
        #    (info.get("description") or info.get("summary") or "").strip()
        #    for info in flat_spec.get("paths", {}).values()
        #]
        #descrizioni = [d for d in descrizioni if d]  # elimina voci vuote

        #if descrizioni:
        #    dynamic_footer += "\n---\n**DESCRIZIONI ENDPOINT DISPONIBILI**\n"
        #    for d in descrizioni:
        #        dynamic_footer += f"- {d}\n"

    # 0.3 – System‑message effettivo
    effective_system_message = system_message.rstrip() + dynamic_footer
    print(effective_system_message)
    # 3.1  LLM ----------------------------------------------------------------
    llm = ChatOpenAI(
        openai_api_key=openai_api_key,
        model_name=model_name,
        streaming=True,
        verbose=verbose,
        **(model_kwargs or {})
    )

    # 3.2  RequestsWrapper configurato con l’host, se presente ---------------
    requests_wrapper = RequestsWrapper()
    #if server_url:
        #requests_wrapper.base_url = server_url

    # 3.3  Strumenti HTTP (GET/POST/PUT/PATCH/DELETE) -------------------------
    default_tools = _default_http_tools(requests_wrapper)

    # 3.4  Strumenti da spec OpenAPI (opzionale) ------------------------------
    spec_tools: List[StructuredTool] = []
    if openapi_spec:
        raw_spec   = load_spec(openapi_spec)
        flat_spec  = dereference_spec(raw_spec)
        spec_tools = [
            make_list_endpoints_tool(flat_spec),
            #make_endpoint_search_tool(flat_spec),
            make_endpoint_details_tool(flat_spec),
        ]

    # 3.5  Merge di tutti gli strumenti ---------------------------------------
    all_tools = default_tools + spec_tools + (extra_tools or [])
    #print(effective_system_message)
    # 3.6  Prompt -------------------------------------------------------------
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", effective_system_message),
            MessagesPlaceholder("chat_history"),
            HumanMessagePromptTemplate.from_template("{input}"),
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # 3.7  Agente + executor --------------------------------------------------
    agent = create_tool_calling_agent(llm, all_tools, prompt)
    executor = AgentExecutor(agent=agent, tools=all_tools, verbose=verbose).with_config(
        {"run_name": "Agent"}
    )
    return executor



# --------------------------------------------------------------------------- #
# 4. Esempio d’uso con tool personalizzato e streaming eventi                 #
# --------------------------------------------------------------------------- #

# 4.1  Tool di esempio -------------------------------------------------------

class OpenApiAgenticTool:
    def __init__(self,
                 openapi_spec: str = "http://127.0.0.1:8000/openapi.json", #"./enac_api.json",      # oppure dict o stringa
                 server_url: str = "http://127.0.0.1:8000",
                 model_name: str = "gpt-4o",
                 openai_api_key: str = "",
                 model_kwargs: dict = {},
                 visible_tools: list[str] | None = None,  # nomi dei tool di cui stampare gli eventi
                 tool_name: str | None = None,
                 tool_description: str | None = None,
                 description_merge_strategy: str = "replace",  # "replace" | "prepend" | "append"
                 ) -> None:

        """
        • Istanzia l’agente con il tool custom.
        • Esegue una query che richiede la moltiplicazione.
        • Stampa in tempo reale gli eventi (LLM tokens, chiamate tool, ecc.).
        """

        self.agent_executor = build_gpt4o_agent(verbose=False,
                                           openapi_spec=openapi_spec, #"./enac_api.json",      # oppure dict o stringa
                                           server_url=server_url,
                                           model_name=model_name,
                                           openai_api_key=openai_api_key,
                                           model_kwargs=model_kwargs)

        # ─────────────────────────────────────────────────────────────
        # 1️⃣  Carica/deriva nome e descrizione del tool
        # ─────────────────────────────────────────────────────────────
        self._tool_name = tool_name
        self._tool_description = tool_description
        self._visible_tools = [t.strip() for t in visible_tools] if visible_tools else \
            ["requests_get","requests_post","requests_put","requests_patch","requests_delete"]

        # Strategia merge descrizione -----------------------
        valid_strategies = {"replace", "prepend", "append"}
        if description_merge_strategy.lower() not in valid_strategies:
            raise ValueError(
                f"description_merge_strategy deve essere una tra {valid_strategies}"
            )
        self._merge_strategy = description_merge_strategy.lower()


        # ------------------------------------------------------------------
        # 1️⃣ recupero title/description dallo spec (se serve)
        # ------------------------------------------------------------------
        spec_title = spec_desc = None
        if openapi_spec:
            try:
                _raw_spec   = load_spec(openapi_spec)
                info_block  = _raw_spec.get("info", {}) if isinstance(_raw_spec, dict) else {}
                spec_title  = info_block.get("title")
                spec_desc   = info_block.get("description")
            except Exception:
                pass   # spec malformato ➜ ignoro

        # Nome: priorità user ▸ spec ▸ default
        self._tool_name = (tool_name or spec_title or "generate_response").strip()

        # Descrizione: fusione secondo strategia
        user_desc  = (tool_description or "").strip() or None
        spec_desc  = (spec_desc or "").strip() or "ERROR: IMPOSSIBLE TO RETRIEVE API INFORMATION!"

        spec_desc = (
                "Provide requests to the following agent that specializes in executing API requests to use the following service:\n"
                + spec_desc)

        merged = None

        if self._merge_strategy == "replace":
            merged = user_desc or spec_desc
        elif self._merge_strategy == "prepend":
            merged = " ".join(filter(None, [user_desc, spec_desc])) or None
        elif self._merge_strategy == "append":
            merged = " ".join(filter(None, [spec_desc, user_desc])) or None

        self._tool_description = merged or "Esegue una richiesta all’agente e restituisce la risposta."

    # ------------------------------------------------------------
    # Helper: mostrare o meno l’evento di un tool
    # ------------------------------------------------------------
    def _show_event_for(self, tool_name: str) -> bool:
        if not self._visible_tools:  # nessuna lista fornita ➜ skip
            return False
        return tool_name in self._visible_tools

    async def generate_streamed_response(self, query: Dict[str, Any], inference_kwargs: Dict[str, Any], stream_only_content: bool = False):

        async for event in self.agent_executor.astream_events(
            query,
            version="v1",
            **inference_kwargs,
        ):
            #print(event)
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
                    print(content, end="")
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



    def generate_response(
            self,
            #query: Dict[str, Any],
            input: str,
            #inference_kwargs: Dict[str, Any] = {},
            stream_only_content: bool = False
    ) -> str:
        """
        Versione sincrona che raccoglie tutti gli output degli eventi
        dalla versione async (astream_events) e li ritorna come stringa.
        """
        output_builder = []

        async def _collector():
            async for event in self.agent_executor.astream_events(
                    #query,
                    {"input": input, "chat_history": []},
                    version="v1",
                    # **inference_kwargs,
            ):
                kind = event.get("event")
                name = event.get("name")
                data = event.get("data", {}) or {}

                if kind == "on_chain_start" and name == "Agent":
                    msg = f"Starting agent: {name} with input: {data.get('input')}\n"
                    output_builder.append(msg)

                elif kind == "on_chain_end" and name == "Agent":
                    result = (data.get("output") or {}).get("output")
                    msg = f"\n--\nDone agent: {name} with output: {result}\n"
                    output_builder.append(msg)

                elif kind == "on_chat_model_stream":
                    chunk = data.get("chunk")
                    content = getattr(chunk, 'content', None)
                    if content:
                        output_builder.append(content)

                elif kind == "on_tool_start":
                    msg = f"\n--\nStarting tool: {name} with inputs: {data.get('input')}\n"
                    if self._show_event_for(event["name"]):
                        output_builder.append(msg)
                        output_builder.append(json.dumps(event))
                        output_builder.append("\n")

                elif kind == "on_tool_end":
                    if self._show_event_for(event["name"]):
                        msg = f"Done tool: {name}\nTool output was: {data.get('output')}\n--\n"
                        output_builder.append(msg)
                        safe_event = _sanitize_event(event)
                        output_builder.append(json.dumps(safe_event))
                        output_builder.append("\n")

            return ''.join(output_builder)

        # Esegui il collector async e ritorna la stringa
        return asyncio.run(_collector())

    def get_tools(self) -> List[StructuredTool]:
        """
        Restituisce un solo tool: generate_response.
        """
        return [
            StructuredTool(
                name=self._tool_name,
                func=self.generate_response,
                description=self._tool_description,
                args_schema=GenerateResponseInput,
            )
        ]

def main():
    agentictool = OpenApiAgenticTool(
        openai_api_key=".....",
        visible_tools = ["requests_get","requests_post","requests_put","requests_patch","requests_delete"],
        tool_name=None,
        tool_description="",
        description_merge_strategy="append"
    )

    # generate_response(chain=agent_executor, query={"input": user_question, "chat_history": []},inference_kwargs={})
    # ** QUESTO** cicla davvero il generator e lo stampa
    user_question = "ottieni la lsita dei cleitni per user id 'user_123', poi ottieni i dettalgi per ciascun cliente"
    #async for _ in agentictool.generate_streamed_response(
    #        query={"input": user_question, "chat_history": []},
    #        inference_kwargs={}
    #):
    #    pass

    resposne = agentictool.generate_response(
            #query={"input": user_question, "chat_history": []},
            #inference_kwargs={}
            input=user_question
         )


    print(resposne)
    print(agentictool.get_tools())

if __name__ == "__main__":
    #asyncio.run(main())
    main()