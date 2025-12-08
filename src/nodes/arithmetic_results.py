from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .. import get_openai_llm, get_gemini_llm
from ..states import MathState

llm = get_gemini_llm()

result_node_template = """
    You are an agent specialized in displaying the results of a mathematical operation.

    you receive the following data in TOON format.
    
    for example.
    
    operation: sum
    result: 11
    
    where operation is -> sum | sub | div | mult
    and result contains the result of the operation
    
    You must return a message that says: The operation sum obtained the following result: 11
    
    examples:
    
    example 1
    operation: sum
    result: 15
    
    message -> The operation sum obtained the following result: 15
    
    example 2
    operation: mult
    result: 32
    
    message -> The operation multiplication obtained the following result: 32
    
    the operation is:
    
    {result}
"""


def result_aritm_node(state: MathState) -> MathState:
    """arithmetic operation"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", result_node_template),
            ("human", "{result}"),
        ]
    )
    chain = prompt | llm
    prediction = chain.invoke(
        {
            "result": state["result"],
        }
    )

    return {
        "messages": state.get("messages", []) + [prediction],
        "message_result": prediction.content
    }


error_node_template = """
    You are an agent who tells the user that you did not understand what they meant.
"""
def error_aritm_node(state: MathState) -> MathState:
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
        "message_result": prediction.content
    }