from __future__ import annotations

from toon import decode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .. import get_openai_llm, get_gemini_llm
from ..states import MathState
from ..tools import arithmetic_tools
from ..utils import build_safe_window

llm = get_gemini_llm()
llm_tools = llm.bind_tools(arithmetic_tools(), tool_choice="auto")

def aritm_operations_tools(state: MathState) -> MathState:
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "aritm_tools"

    return "result_node"

op_node_template = """
    You are an expert at performing arithmetic operations using a tool called arithmetic_operation.

    The tool receives:
      operation: str
      operand_a: int
      operand_b: int
    
    and returns a dict:
      {{ "operation": "operation", "result": number }}
    
    Allowed operations: sum | sub | div | mult.
    
    - If you DO NOT see any tool result yet in the conversation, you MAY call the tool.
    - If you ALREADY see the tool result in previous messages, DO NOT call the tool again.
      Instead, answer in TOON format:
        operation: <operation>
        result: <number>
    
    The user operation is:
    {operation}  
"""
def op_aritm_node(state: MathState) -> MathState:
    """op operation"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", op_node_template),
            MessagesPlaceholder("messages"),
            ("human", "{operation}"),
        ]
    )
    chain = prompt | llm_tools
    messages = build_safe_window(state.get("messages", []))
    prediction = chain.invoke(
        {
            "operation": state["operation"],
            "messages": messages
        }
    )

    return {
        "messages": state.get("messages", []) + [prediction],
        "result" :  decode(prediction.content) if prediction.content != '' else ''
    }


