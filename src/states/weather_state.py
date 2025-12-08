from langgraph.graph import MessagesState

class WeatherState(MessagesState):
    """Estado compartido entre nodos de un agente."""

    question : str
    result: str
    state: str
