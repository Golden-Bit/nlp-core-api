import asyncio
from typing import Any, List

from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from chains.chain_scripts.utilities.dataloader import DocumentToolKitManager
from chains.chain_scripts.utilities.graph import GraphManager
from chains.chain_scripts.utilities.mongodb import MongoDBToolKitManager
from chains.chain_scripts.utilities.noop import NoopToolKitManager
from chains.chain_scripts.utilities.vectorstore import VectorStoreToolKitManager
from chains.chain_scripts.utilities.report import TemplateManager
#from chains.chain_scripts.utilities.oepnapi_agent import OpenApiAgenticTool

# Mapping degli strumenti
tools_map = {
    "MongoDBTools": MongoDBToolKitManager,
    "DocumentTools": DocumentToolKitManager,
    "VectorStoreTools": VectorStoreToolKitManager,
    "TemplateManager": TemplateManager,
    "GraphManager": GraphManager,
    "NoopTool": NoopToolKitManager,  # <-- NEW
    #"OpenApiAgenticTool": OpenApiAgenticTool
}

def get_chain(llm: Any = None,
              system_message: str = "You are a helpful assistant",
              tools: List[Any] = None):
    """
    Crea una chain utilizzando `create_tool_calling_agent`.

    Args:
        llm: Il modello LLM da utilizzare.
        system_message: Messaggio del sistema per il contesto del prompt.
        tools: Lista di strumenti da integrare.

    Returns:
        AgentExecutor configurato.
    """
    agent_tools = []
    for tool in tools:
        initialized_tools = tools_map[tool["name"]](**tool["kwargs"]).get_tools()
        agent_tools.extend(initialized_tools)

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder("chat_history"),  # lista di BaseMessage
            MessagesPlaceholder("input"),  # UN singolo HumanMessage
            MessagesPlaceholder("agent_scratchpad"),
        ]
    )

    # Crea l'agente utilizzando create_tool_calling_agent
    agent = create_tool_calling_agent(llm, agent_tools, prompt)

    # Configura l'executor
    agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True).with_config(
        {"run_name": "Agent"}
    )
    return agent_executor
