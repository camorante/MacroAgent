from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from toon import decode

from src import get_openai_llm
from src.states import MathState

llm = get_openai_llm()

def start_main_route(state: MathState) -> str:
    decision = decode(state["decision"]) if state["decision"] != '' else ''
    if decision['operation'] == 'trigonometric':
        return 'to_trigonometric'
    elif decision['operation'] == 'arithmetic':
        return 'to_arithmetic'
    else:
        return 'to_error'

op_node_template =  """
    You are an agent specialized in analyzing what type of mathematical operation a user requests. 
    The operation can only be a trigonometric or arithmetic operation.

    Return a TOON with information about the type of operation, like this ->

    operation: trigonometric | arithmetic | error

    If you do not understand the operation indicated by the user, return the following TOON ->
    
    operation: error

    example 1
    Question -> What is the sine of 60 degrees?.
    Response -> 
    operation: trigonometric

    example 2
    Question -> calculate the sum of 4 + 6
    Response -> 
    operation: arithmetic

    example 3
    Question -> hey how is it going?
    Response -> 
    operation: error
    {question}
"""
def main_route_analyzer_node(state: MathState) -> MathState:
    """op analyzer"""

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", op_node_template),
            ("human", "{question}"),
        ]
    )
    chain = prompt | llm
    prediction = chain.invoke(
        {
            "question": state["question"],
            "messages": state.get("messages", [])
        }
    )

    return {
        "messages": state.get("messages", []) + [prediction],
        "decision": prediction.content
    }

error_node_template = """
    You are an agent who tells the user that you did not understand what they meant.
"""
def main_error_node(state: MathState) -> MathState:
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