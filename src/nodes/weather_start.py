from __future__ import annotations
from toon import decode
from langchain_core.prompts import ChatPromptTemplate

from .. import get_openai_llm
from ..states import WeatherState

llm = get_openai_llm()

def start_weather_route(state: WeatherState) -> WeatherState:
    decision = decode(state["state"])

    if decision['state'] == 'error':
        return 'to_error'
    else:
        return 'to_weather'

start_node_template = """
    You are an agent that ONLY detects whether alerts the user is asking about the weather
    in a specific U.S. state and outputs a normalized state code.
    
    The user may write in English or Spanish and may have minor typos.
    Your job is to map whatever they say to one of the official two-letter U.S. state 
    or territory codes.
    
    Valid state codes are:
    ["AL","AK","AS","AR","AZ","CA","CO","CT","DE","DC","FL","GA","GU","HI","ID","IL","IN","IA","KS","KY","LA",
    "ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ","NM","NY","NC","ND","OH","OK","OR","PA","PR",
    "RI","SC","SD","TN","TX","UT","VT","VI","VA","WA","WV","WI","WY","MP","PW","FM","MH"]
    
    INSTRUCTIONS:
    
    1. Decide if the user is asking for the WEATHER in a specific U.S. state or territory.
       - The user might use Spanish (e.g. "estado de", "clima", "tiempo").
       - The user might misspell the state name (e.g. "floria" -> "Florida" -> FL).
    
    2. If they ARE asking for the weather alert in ONE specific state, output EXACTLY:
       state: <STATE_CODE>
       where <STATE_CODE> is one of the valid codes above, Otherwise, mark error.
    
    3. If you cannot confidently map the user's message to a single state,
       or if they are NOT asking about the weather in a U.S. state, output:
       state: null
    
    4. Output ONLY the line with "state: ...". No explanations, no extra text.
    
    EXAMPLES:
    
    User: "What's the weather in California today?"
    Assistant: 
    state: CA
    
    User: "Necesito saber el clima del estado de la floria"
    Assistant:
    state: FL
    
    User: "¿Cómo está el tiempo en Nueva York?"
    Assistant:
    state: NY
    
    User: "Quiero viajar, ¿qué tal está el clima en Texas y en Florida?"
    Assistant:
    state: Error
    
    User: "Quiero saber el clima de mi ciudad, Bogotá"
    Assistant:
    state: Error

    {question}
"""
def start_weather_node(state: WeatherState) -> WeatherState:
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", start_node_template),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    prediction = chain.invoke(
        {
            "question": state["question"],
        }
    )

    return {
        "messages": state.get("messages", []) + [prediction],
        "state": prediction.content
    }

error_node_template = """
    You are an agent who tells the user that you did not understand what they meant.
"""
def error_weather_node(state: WeatherState) -> WeatherState:
    """error operation"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", error_node_template)
        ]
    )
    chain = prompt | llm
    prediction = chain.invoke({})

    return {
        **state,
        "messages": state.get("messages", []) + [prediction],
        "result": prediction.content
    }