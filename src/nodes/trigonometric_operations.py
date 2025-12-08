from __future__ import annotations

from langchain_core.tools import tool
from toon import decode
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from .. import get_openai_llm
from ..states import MathState
from ..tools import trigonometric_tools

llm = get_openai_llm()
llm_tools = llm.bind_tools(trigonometric_tools(), tool_choice="auto")

def trig_operations_tools(state: MathState) -> MathState:
    msg = state["messages"][-1]
    if msg.tool_calls:
        return "trig_tools"

    return "result_node"


op_node_template = """
    You are an expert at performing trigonometric operations using a tool called trigonometric_operations.

    The tool receives:
      operation: str
      operand: float

    and the tool returns a dict:
      {{ "operation": "operation", "result": number }}

    Allowed operations: cos | sin | tan

    - If you DO NOT see any tool result yet in the conversation, you MAY call the tool.
    - If you ALREADY see the tool result in previous messages, DO NOT call the tool again.
      Instead, answer in TOON format:
        operation: <operation>
        result: <number>

    The user operation is:
    {operation}  
"""

def op_trig_node(state: MathState) -> MathState:
    """op operation"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", op_node_template),
            MessagesPlaceholder("messages"),
            ("human", "{operation}"),
        ]
    )
    chain = prompt | llm_tools
    messages = state.get("messages", [])[-2:]
    prediction = chain.invoke(
        {
            "operation": state["operation"],
            "messages": messages
        }
    )

    return {
        "messages": state.get("messages", []) + [prediction],
        "result": decode(prediction.content) if prediction.content != '' else ''
    }