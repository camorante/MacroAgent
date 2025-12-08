from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate

from .. import get_openai_llm
from ..states import MathState

llm = get_openai_llm()

result_node_template = """
    You are an agent specialized in displaying the results of a trigonometric operation(cosine, sine, tangent).

    you receive the following data in TOON format.

    for example.

    operation: cos
    result: 0.5

    where operation is -> cos | sin | tan 
    and result contains the result of the operation

    You must return a message that says: The operation cosine obtained the following result: 0.5

    examples:

    example 1
    operation: sin
    result: 0.9848077530

    message -> The operation sine obtained the following result: 0.9848077530

    example 2
    operation: tan
    result: 1.7320508

    message -> The operation tangent obtained the following result: 1.7320508

    the operation is:
    {result}
"""


def result_trig_node(state: MathState) -> MathState:
    """trigonometric operation"""

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

def error_trig_node(state: MathState) -> MathState:
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