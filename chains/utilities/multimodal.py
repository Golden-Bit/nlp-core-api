# chains/utilities/multimodal.py
from typing import List, Dict, Any, Optional
from langchain_core.messages import HumanMessage, AIMessage, BaseMessage

def build_parts(text: Optional[str], images: Optional[List[Dict[str, Any]]]) -> List[Dict[str, Any]]:
    """Costruisce la lista di content‑parts testo+immagini."""
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for image in images or []:
        parts.append(image)
    return parts

def build_parts_legacy(text: Optional[str], images: Optional[List[str]]) -> List[Dict[str, Any]]:
    """Costruisce la lista di content‑parts testo+immagini."""
    parts = []
    if text:
        parts.append({"type": "text", "text": text})
    for url in images or []:
        parts.append({"type": "image_url", "image_url": {"url": url, "detail": "auto"}})
    return parts

def to_message(obj: Dict[str, Any]) -> BaseMessage:
    """Converte un item history → BaseMessage (Human/AI) preservando le parts."""
    role   = obj["role"]
    parts  = obj["parts"]
    return HumanMessage(content=parts) if role == "user" else AIMessage(content=parts)
