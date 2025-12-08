from __future__ import annotations

from langchain_mcp_adapters.tools import load_mcp_tools
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from mcp import stdio_client, StdioServerParameters, ClientSession

from src import get_openai_llm, get_checkpoint_store, get_async_checkpoint_store
from src.nodes import start_weather_node, start_weather_route, make_get_weather_node, error_weather_node, \
    get_weather_mcp_tools
from src.states import WeatherState

llm = get_openai_llm()

server_params = StdioServerParameters(
    command="python",
    args=["src/mcp/weather_server.py"],
)
async def build_weather_graph(user_input: str, show_graph = False):
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            tools = await load_mcp_tools(session)
            model_with_tools = llm.bind_tools(tools)
            weather_tool_node = ToolNode(tools)

            get_weather_node = make_get_weather_node(model_with_tools)

            graph_builder = StateGraph(WeatherState)
            graph_builder.add_node("start_weather_node", start_weather_node)
            graph_builder.add_node("error_weather_node", error_weather_node)
            graph_builder.add_node("get_weather_node", get_weather_node)
            graph_builder.add_node("weather_tools", weather_tool_node)

            graph_builder.add_edge(START, "start_weather_node")
            graph_builder.add_conditional_edges(
                "start_weather_node",
                start_weather_route,
                {
                    "to_error": "error_weather_node",
                    "to_weather": "get_weather_node"
                }
            )

            graph_builder.add_conditional_edges("get_weather_node", get_weather_mcp_tools, ["weather_tools", END])
            graph_builder.add_edge("weather_tools", "get_weather_node")
            graph_builder.add_edge("error_weather_node", END)
            checkpointer = await get_async_checkpoint_store()
            agent = graph_builder.compile(checkpointer=checkpointer)

            if show_graph:
                print(agent.get_graph().print_ascii())

            config = {
                "configurable": {
                    "user_id": "1",
                    "thread_id": "1",
                }
            }

            final_state = await agent.ainvoke(
                {"messages": [("user", user_input)], "question": user_input},
                config=config,
            )

            return final_state

