import asyncio
from typing import Any, List

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain.tools import tool
from langchain_core.callbacks import Callbacks
from langchain_core.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, MessagesPlaceholder
from langchain_openai import ChatOpenAI

from chains.chain_scripts.utilities.dataloader import DocumentToolKitManager
from chains.chain_scripts.utilities.mongodb import MongoDBToolKitManager
from chains.chain_scripts.utilities.vectorstore import VectorStoreToolKitManager


tools_map = {
    "MongoDBTools": MongoDBToolKitManager,
    "DocumentTools": DocumentToolKitManager,
    "VectorStoreTools": VectorStoreToolKitManager
}


def get_chain(llm: Any = None,
              system_message: str = "you are an helpful assistant",
              tools: List[Any] = None):

    agent_tools = []
    for tool in tools:
        initialized_tools = tools_map[tool["name"]](**tool["kwargs"]).get_tools()
        agent_tools.extend(initialized_tools)

    # Get the prompt to use - you can modify this!
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_message),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}"),
            MessagesPlaceholder(variable_name="agent_scratchpad"),  # Placeholder richiesto per i passaggi intermedi
        ]
    )

    agent = create_openai_tools_agent(
        llm.with_config({"tags": ["agent_llm"]}), agent_tools, prompt
    )
    agent_executor = AgentExecutor(agent=agent, tools=agent_tools, verbose=True).with_config(
        {"run_name": "Agent"}
    )
    return agent_executor

