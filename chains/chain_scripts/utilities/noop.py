# chains/chain_scripts/utilities/noop.py
from __future__ import annotations

from typing import Any, Optional
from pydantic import BaseModel, Field
try:
    # Pydantic v2
    from pydantic import ConfigDict
    _HAS_V2 = True
except Exception:
    _HAS_V2 = False

from langchain_core.tools import StructuredTool


# --------------------------------------------------------------------------- #
# Pydantic schema (string-only, tollerante ai campi extra)
# --------------------------------------------------------------------------- #
class NoopArgs(BaseModel):
    """Placeholder args: opzionale 'reason', tutti gli altri campi (se stringhe) sono ignorati."""
    reason: Optional[str] = Field(
        default=None,
        description="Motivo puramente informativo/di debug. Verrà ignorato."
    )

    if _HAS_V2:
        model_config = ConfigDict(extra="allow")  # pydantic v2
    else:
        class Config:  # pydantic v1
            extra = "allow"


# --------------------------------------------------------------------------- #
# Manager stile Toolkit: espone un StructuredTool “di cortesia”
# --------------------------------------------------------------------------- #
class NoopToolKitManager:
    """
    Bridge per creare un tool segnaposto. Non fa nulla; serve solo quando
    l'agente richiede almeno un tool disponibile.
    """

    def __init__(self, message: str | None = None) -> None:
        # Messaggio base restituito dal tool (personalizzabile via kwargs)
        self._message = (
            message
            or "NOOP_TOOL: placeholder non operativo. Ignora tutti i parametri e non deve essere usato."
        )

    # Funzione invocata dal tool: non fa nulla
    def _noop(self, reason: Optional[str] = None, **kwargs: Any) -> str:
        return self._message

    # Factory StructuredTool (stessa impostazione del tuo vectorstore_toolkit)
    def get_tools(self):
        return [
            StructuredTool(
                name="NoopTool",
                func=self._noop,
                description=(
                    "⚠️ NO-OP PLACEHOLDER TOOL — NON OPERATIVO.\n"
                    "Questo strumento esiste solo per soddisfare agenti che richiedono almeno un tool. "
                    "Non esegue alcuna azione e **non va usato** se ci sono altri strumenti disponibili."
                ),
                args_schema=NoopArgs,
            )
        ]
