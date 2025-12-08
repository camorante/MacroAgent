import asyncio
import sys

from langchain_core.messages import AIMessage

from src.orchestrator import build_main_agent
from src.agents.weather import  build_weather_graph

if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

def main():
    print("=== Demo LangGraph multi-agente ===")
    print("Escribe una pregunta y pulsa Enter. Ctrl+C para salir.\n")
    agent = build_main_agent
    is_async = False
    answer = ''
    while True:
        try:
            user_input = input("Tú: ")
        except (EOFError, KeyboardInterrupt):
            print("\nHasta luego 👋")
            break
        if user_input == '\\bye':
            break
        elif user_input == '\\weather':
            agent = build_weather_graph
            continue
        elif user_input == '\\main':
            agent = build_main_agent
            is_async = False
            continue
        if is_async:
            answer = asyncio.run(run_once_async(user_input, agent))
        else:
            answer = run_once(user_input, agent)
        print(f"\nAsistente:\n{answer}\n")

async def run_once_async(user_input: str, agent) -> str:
    final_state = await build_weather_graph(user_input, show_graph=True)

    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else "No se generó respuesta."

def run_once(user_input: str, agent) -> str:
    graph = agent(show_graph=True)

    config = {"configurable_id": "1", "configurable": {"user_id": "1", "thread_id": "1"}}

    final_state = graph.invoke({"messages": [("user", user_input)], "question": user_input}, config=config)

    ai_messages = [m for m in final_state["messages"] if isinstance(m, AIMessage)]
    return ai_messages[-1].content if ai_messages else "No se generó respuesta."

if __name__ == "__main__":
    main()