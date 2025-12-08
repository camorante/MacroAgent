from __future__ import annotations

from langchain_core.language_models import BaseChatModel
from langgraph.constants import END
from mcp import ClientSession, stdio_client
from toon import decode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .. import get_openai_llm
from ..states import WeatherState

def get_weather_mcp_tools(state: WeatherState) -> WeatherState:
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "weather_tools"

    return END

weather_get_node_template = """
    You are an expert at checking the weather alert using an MCP server and an MCP tool called get_alerts.
    
    You will receive the following data in TOON format:
    
    state: <STATE_CODE>
    
    here <STATE_CODE> is one of the valid codes below
    
    Valid state codes are:
    ["AL","AK","AS","AR","AZ","CA","CO","CT","DE","DC","FL","GA","GU","HI","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","PR",
    "RI","SC","SD","TN","TX","UT","VT","VI","VA","WA","WV","WI","WY","MP","PW","FM","MH"]
    
    returns the data obtained to the user in a friendly and attentive manner
    
    The country state is:
    {state}
    
"""

def make_get_weather_node(model_with_tools: BaseChatModel):
    def get_weather_node(state: WeatherState) -> WeatherState:
        prompt = ChatPromptTemplate.from_messages(
            [
                ("system", weather_get_node_template),
                MessagesPlaceholder("messages"),
                ("human", "{state}"),
            ]
        )
        chain = prompt | model_with_tools
        prediction = chain.invoke(
            {
                "state": state["state"],
                "messages": state.get("messages", [])[-2:]
            }
        )
        return {
            "messages": state.get("messages", []) + [prediction],
            "result": prediction.content
        }

    return get_weather_node
