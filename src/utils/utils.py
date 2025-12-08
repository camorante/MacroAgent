from typing import List

from langchain_core.messages import ToolMessage, AIMessage, HumanMessage


def build_safe_window(
    messages: List,
    max_messages: int = 8,
):
    """
    Devuelve una ventana de historial que:
    - No empieza en AI con tool_calls.
    - No empieza en ToolMessage.
    - Idealmente empieza en HumanMessage.
    - Puede descartar interacciones viejas para ahorrar tokens.
    """

    # 1) Tomamos solo los últimos N mensajes
    window = list(messages[-max_messages:])

    # 2) Asegurar que la conversación **no empiece** en:
    #    - AI con tool_calls
    #    - ToolMessage
    #    Porque eso rompe la secuencia que Gemini espera
    while window:
        first = window[0]

        # Si empieza en Tool o en AI con tool_calls, lo tiramos
        if isinstance(first, ToolMessage):
            window.pop(0)
            continue

        if isinstance(first, AIMessage) and getattr(first, "tool_calls", None):
            window.pop(0)
            continue

        # Si llega aquí, el primero ya es seguro (Human, AI sin tools, etc.)
        break

    # 3) Asegurarnos (por seguridad extra) que el primero sea HumanMessage
    #    Si quieres permitir SystemMessage, puedes adaptarlo.
    while window and not isinstance(window[0], HumanMessage):
        window.pop(0)

    return window