from langgraph.graph import MessagesState
from typing import Annotated, Sequence, Literal

class MathState(MessagesState):
    """Estado compartido entre nodos de un agente."""

    question : str
    result: str
    operation: str
    message_result: str
    decision: str