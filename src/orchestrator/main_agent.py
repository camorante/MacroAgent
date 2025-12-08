from langgraph.checkpoint.memory import MemorySaver
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from src import get_checkpoint_store
from src.agents.math import build_trigonometric_graph, build_arithmetic_graph
from src.nodes import main_route_analyzer_node, start_main_route, main_error_node
from src.states import MathState

arithmetic_graph = build_arithmetic_graph()
trigonometric_graph = build_trigonometric_graph()

def build_main_agent(show_graph = False):
    graph_builder = StateGraph(MathState)
    graph_builder.add_node("main_route_analyzer_node", main_route_analyzer_node)
    graph_builder.add_node("arithmetic_graph", arithmetic_graph)
    graph_builder.add_node("trigonometric_graph",trigonometric_graph)
    graph_builder.add_node("main_error_node", main_error_node)

    graph_builder.add_edge(START, "main_route_analyzer_node")

    graph_builder.add_conditional_edges(
        "main_route_analyzer_node",
        start_main_route,
        {
            "to_error": "main_error_node",
            "to_arithmetic": "arithmetic_graph",
            "to_trigonometric": "trigonometric_graph",
        }
    )
    graph_builder.add_edge("arithmetic_graph", END)
    graph_builder.add_edge("trigonometric_graph", END)
    graph_builder.add_edge("main_error_node", END)

    checkpointer = get_checkpoint_store()
    agent = graph_builder.compile(checkpointer=checkpointer)
    if show_graph:
        print(agent.get_graph().print_ascii())
    return agent
