from __future__ import annotations
from toon import decode
from langchain_core.prompts import ChatPromptTemplate

from .. import get_openai_llm
from ..states import MathState

llm = get_openai_llm()

start_node_template = """
    You are an agent specialized in analyzing what type of trigonometric operation a user is requesting. 
    The operation can only be cosine, sine and tangent and can only involve one number in degrees.

    Return a TOON with information about the operation and the numbers as degree, like this ->

    operation: cos | sin | tan | error
    operand: degrees

    If you do not understand the operation indicated by the user, return the following TOON ->
    
    operation: error
    operand: 0

    example 1
    Question -> What is the sine of 60 degrees?.
    Response -> 
    operation: sin
    operand: 60

    example 2
    Question -> calculate the cosine of 120
    Response -> 
    operation: cos
    operand: 120

    {question}
"""

def start_trig_route(state: MathState) -> MathState:
    decision = decode(state["operation"])

    if decision['operation'] == 'error':
        return 'to_error'
    else:
        return 'to_operations'

def start_trig_node(state: MathState) -> MathState:
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
        "operation": prediction.content
    }