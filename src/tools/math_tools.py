from langchain_core.tools import tool
import math

@tool
def arithmetic_operations(operation: str, operand_a: int, operand_b: int) -> dict:
    """tools that can perform basic arithmetic operations such as addition, subtraction, division and multiplication"""

    if operation == 'sum':
        return { "operation": 'sum', "result": operand_a + operand_b}
    elif  operation == 'sub':
        return { "operation": 'sub', "result": operand_a - operand_b}
    elif  operation == 'div':
        return { "operation": 'div', "result":  (0 if operand_b == 0 else operand_a / operand_b)}
    elif  operation == 'mult':
        return { "operation": 'mult', "result": operand_a * operand_b}
    else:
        return {"operation": '', "result": 0}

@tool
def trigonometric_operations(operation: str, degrees: float):
    """tools that can perform basic trigonometric operations such as cosine, sine, and tangent"""
    if operation == 'cos':
        return {"operation": 'cos', "result": math.cos(math.radians(degrees))}
    elif operation == 'sin':
        return {"operation": 'sin', "result": math.sin(math.radians(degrees))}
    elif operation == 'tan':
        return {"operation": 'tan', "result": math.tan(math.radians(degrees))}
    else:
        return {"operation": '', "result": 0}

def arithmetic_tools():
    return [arithmetic_operations]

def trigonometric_tools():
    return [trigonometric_operations]