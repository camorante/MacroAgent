from __future__ import annotations
from toon import decode
from langchain_core.prompts import ChatPromptTemplate

from .. import get_openai_llm, get_gemini_llm
from ..states import MathState

llm = get_openai_llm()

start_node_template = """
    You are an agent specialized in analyzing what type of mathematical operation a user is requesting. 
    The operation can only be addition, subtraction, division, or multiplication and can only involve two numbers.
    
    Return a TOON with information about the operation and the two numbers as operands, like this.
    
    operation: sum | sub | div | mult | error
    operand_a: operand A
    operand_b: operand B
    
    If you do not understand the operation indicated by the user, return the following TOON
    operation: error
    operand_a: 0
    operand_b: 0
    
    example 1
    Question -> Add 8 plus 3, please.
    Response -> 
    operation: sum
    operand_a: 8
    operand_b: 3
    
    example 2
    Question -> multiply the numbers 563 by -36
    Response -> 
    operation: mult
    operand_a: 563
    operand_b: -36
        
    {question}
"""

def start_aritm_route(state: MathState) -> MathState:
    decision = decode(state["operation"])

    if decision['operation'] == 'error':
        return 'to_error'
    else:
        return 'to_operations'

def start_aritm_node(state: MathState) -> MathState:
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