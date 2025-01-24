import base64
from typing import List, Optional, Union, Callable
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain.schema.messages import SystemMessage, HumanMessage
from langchain_core.messages import AIMessage


class LLMFunctionBase:
    """
    Base class for defining LLM-based functions with attributes like name, description,
    system message, OpenAI API key, and message history. Supports optional postprocessing of output.
    """
    def __init__(
        self,
        name: str,
        description: str,
        system_message: str,
        openai_api_key: str,
        temperature: float = 0.5,
        max_tokens: int = 2048,
        postprocess: Optional[Callable[[str], str]] = None
    ):
        self.name = name
        self.description = description
        self.system_message = system_message
        self.openai_api_key = openai_api_key
        self.messages: List[Union[SystemMessage, HumanMessage, AIMessage]] = []
        self.postprocess = postprocess
        self.chat_model = ChatOpenAI(
            model="gpt-4o",
            temperature=temperature,
            max_tokens=max_tokens,
            openai_api_key=openai_api_key
        )

    def execute(self, new_messages: List[dict]) -> str:
        """
        Executes the LLM with the provided new messages, returning the model's output.
        Applies postprocessing if a postprocess function is defined.
        """
        if not self.messages:
            # Add system message only at the start
            self.messages.append(SystemMessage(content=self.system_message))

        # Convert new_messages into HumanMessage format
        human_message = HumanMessage(content=new_messages)
        self.messages.append(human_message)

        response = self.chat_model(self.messages)
        self.messages.append(AIMessage(content=response.content))

        output = response.content

        # Apply postprocessing if defined
        if self.postprocess:
            output = self.postprocess(output)

        return output
