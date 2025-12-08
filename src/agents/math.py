from __future__ import annotations

from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode

from ..states import MathState
from ..nodes import (
    start_aritm_node,
    op_aritm_node,
    result_aritm_node,
    start_aritm_route,
    error_aritm_node,
    aritm_operations_tools,
    start_trig_node,
    op_trig_node,
    result_trig_node,
    start_trig_route,
    error_trig_node,
    trig_operations_tools,
)
from ..tools import arithmetic_tools, trigonometric_tools

def build_arithmetic_graph(show_graph = False):
    graph_builder = StateGraph(MathState)

    graph_builder.add_node("start_node", start_aritm_node)
    graph_builder.add_node("op_node", op_aritm_node)
    graph_builder.add_node("result_node", result_aritm_node)
    graph_builder.add_node("error_node", error_aritm_node)
    graph_builder.add_node("aritm_tools", ToolNode(arithmetic_tools()))

    graph_builder.add_edge(START, "start_node")
    graph_builder.add_conditional_edges(
        "start_node",
        start_aritm_route,
        {
            "to_error": "error_node",
            "to_operations": "op_node"
        }
    )

    graph_builder.add_conditional_edges("op_node", aritm_operations_tools, ["aritm_tools", "result_node"])
    graph_builder.add_edge("aritm_tools", "op_node")
    graph_builder.add_edge("result_node", END)
    graph_builder.add_edge("error_node", END)
    agent = graph_builder.compile()
    if show_graph:
        print(agent.get_graph().print_ascii())
    return agent

def build_trigonometric_graph(show_graph = False):
    graph_builder = StateGraph(MathState)

    graph_builder.add_node("start_node", start_trig_node)
    graph_builder.add_node("op_node", op_trig_node)
    graph_builder.add_node("result_node", result_trig_node)
    graph_builder.add_node("error_node", error_trig_node)
    graph_builder.add_node("trig_tools", ToolNode(trigonometric_tools()))

    graph_builder.add_edge(START, "start_node")
    graph_builder.add_conditional_edges(
        "start_node",
        start_trig_route,
        {
            "to_error": "error_node",
            "to_operations": "op_node"
        }
    )

    graph_builder.add_conditional_edges("op_node", trig_operations_tools, ["trig_tools", "result_node"])
    graph_builder.add_edge("trig_tools", "op_node")
    graph_builder.add_edge("result_node", END)
    graph_builder.add_edge("error_node", END)
    agent = graph_builder.compile()

    if show_graph:
        print(agent.get_graph().print_ascii())
    return agent