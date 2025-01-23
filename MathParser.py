from lark import Lark, Tree, Token
from graphviz import Digraph
import pandas as pd
import matplotlib.pyplot as plt

# Define the grammar for mathematical expressions
math_grammar = '''
    start: expr
    expr: expr "+" expr -> add
        | expr "-" expr -> sub
        | expr "*" expr -> mul
        | expr "/" expr -> div
        | NUMBER -> number
        | "(" expr ")"
    NUMBER: /[0-9]+/
    %import common.WS
    %ignore WS
'''

# Create the Lark parser using the grammar
parser = Lark(math_grammar, start='start', parser='lalr')

# Function to compute synthesized and inherited attributes
def compute_attributes(tree, inherited_value=None):
    """
    Compute synthesized and inherited attributes for nodes.
    Only non-leaf nodes will have an inherited value.
    """
    if isinstance(tree, Tree):
        op_map = {"add": "+", "sub": "-", "mul": "*", "div": "/"}
        synthesized = None
        inherited = inherited_value  # Use inherited_value only for non-leaf nodes

        if tree.data in op_map:  # Binary operator nodes
            operator = op_map[tree.data]

            # Compute left child
            left_child = compute_attributes(tree.children[0], inherited)

            # Compute right child
            right_child_inherited = left_child["synthesized"]  # Propagate synthesized value as inherited
            right_child = compute_attributes(tree.children[1], right_child_inherited)

            # Perform operation
            lhs = left_child["synthesized"]
            rhs = right_child["synthesized"]

            if operator == "+":
                synthesized = lhs + rhs
            elif operator == "-":
                synthesized = lhs - rhs
            elif operator == "*":
                synthesized = lhs * rhs
            elif operator == "/":
                synthesized = lhs / rhs

            expression = f"({lhs} {operator} {rhs})"
            return {
                "inherited": inherited,
                "synthesized": synthesized,
                "expression": expression
            }

        elif tree.data == "number":  # Leaf nodes
            synthesized = int(tree.children[0])
            expression = str(synthesized)
            return {
                "inherited": None,  # Leaf nodes do not have an inherited value
                "synthesized": synthesized,
                "expression": expression
            }

        elif tree.data == "expr":  # Parentheses rule
            # Propagate the value of the inner expression
            return compute_attributes(tree.children[0], inherited)

    elif isinstance(tree, Token):  # Token case (NUMBER leaf)
        return {"inherited": None, "synthesized": int(tree), "expression": str(tree)}

    return {"inherited": None, "synthesized": None, "expression": None}

# Function to visualize the parse tree
def visualize_tree(parse_tree, raw_expression):
    """
    Creates a visualization of the parse tree using Graphviz.
    Displays the raw input expression in a node directly below the 'start' node.
    """
    dot = Digraph()
    dot.attr(rankdir="TB")  # Top-to-bottom layout

    def add_nodes(dot, tree, parent=None, inherited=None, is_raw_expression=False):
        if not tree:  # Handle None cases
            return

        attributes = compute_attributes(tree, inherited)
        synthesized = attributes.get("synthesized", None)
        expression = attributes.get("expression", "")
        inherited_value = attributes.get("inherited", None)

        # Create a unique node identifier
        node_name = str(id(tree))

        if is_raw_expression:  # Special handling for the raw expression node
            dot.node(
                node_name,
                f"EXPRESSION\n{raw_expression}",
            )
            if parent:
                dot.edge(parent, node_name)

        elif isinstance(tree, Tree):
            # Only display inherited for non-leaf nodes
            inherited_display = f"Inherited: {inherited_value}\n" if inherited_value is not None else ""
            dot.node(
                node_name,
                f"{tree.data.upper()}\n{inherited_display}Expression: {expression}\nSynthesized: {synthesized}",
            )
            if parent:
                dot.edge(parent, node_name)

            # Propagate inherited value explicitly to child nodes
            for i, child in enumerate(tree.children):
                child_inherited = synthesized if i == 1 else inherited  # Propagate only to non-leaf nodes
                add_nodes(dot, child, node_name, child_inherited)

        elif isinstance(tree, Token):  # Leaf nodes
            dot.node(
                node_name,
                f"NUM: {tree}\nExpression: {expression}\nSynthesized: {synthesized}",
            )
            if parent:
                dot.edge(parent, node_name)

    # Add the 'start' node
    start_node_name = str(id(parse_tree)) + "_start"
    dot.node(start_node_name, "START")

    # Add the node displaying the raw input expression
    raw_expression_node_name = str(id(parse_tree)) + "_raw_expression"
    dot.node(raw_expression_node_name, f"EXPRESSION\n{raw_expression}")
    dot.edge(start_node_name, raw_expression_node_name)

    # Add remaining nodes recursively
    for child in parse_tree.children:
        add_nodes(dot, child, raw_expression_node_name)

    return dot

# Function to check for balanced parentheses
def is_balanced(expression):
    stack = []
    for char in expression:
        if char == "(":
            stack.append(char)
        elif char == ")":
            if not stack or stack.pop() != "(":
                return False
    return not stack

# Standalone function to evaluate the expression using BODMAS rules
def evaluate_expression(expression):
    """
    Evaluates a mathematical expression using BODMAS rules.
    """
    precedence = {'+': 1, '-': 1, '*': 2, '/': 2}
    operators = []
    operands = []

    def apply_operator():
        op = operators.pop()
        b = operands.pop()
        a = operands.pop()
        if op == '+':
            operands.append(a + b)
        elif op == '-':
            operands.append(a - b)
        elif op == '*':
            operands.append(a * b)
        elif op == '/':
            operands.append(a / b)

    i = 0
    while i < len(expression):
        char = expression[i]
        if char.isdigit():
            num = ''
            while i < len(expression) and expression[i].isdigit():
                num += expression[i]
                i += 1
            operands.append(int(num))
            continue
        elif char in precedence:
            while (operators and operators[-1] != '(' and
                   precedence[operators[-1]] >= precedence[char]):
                apply_operator()
            operators.append(char)
        elif char == '(':
            operators.append(char)
        elif char == ')':
            while operators and operators[-1] != '(':
                apply_operator()
            operators.pop()
        i += 1

    while operators:
        apply_operator()

    return operands[0]

# Lexer callback function to capture tokenization and display lexer operations
def lexer_callback(token, log_file):
    """
    Capture and log the tokenization process.
    """
    operation = f"Token: {token.type}, Value: {token.value}\n"
    print(operation)  # Display token type and value
    log_file.write(operation)  # Write lexer operation to file

# Function to visualize tokenization
def visualize_tokenization(expression):
    """
    Visualizes the tokenization process by showing the tokens produced by the lexer.
    It also includes a table with Production rules and Semantic Analysis.
    """
    # Tokenize the expression using the lexer
    tokens = list(parser.lex(expression))
    print(f"Tokens: {tokens}")  # Debugging: print the tokens
    
    if not tokens:
        raise ValueError("Lexer produced no tokens. Please check the input expression.")
    
    # Create a log file to capture lexer operations
    with open("lexer_operations_log.txt", "w") as log_file:
        log_file.write("Lexer Operations Log:\n")
        for token in tokens:
            lexer_callback(token, log_file)  # Capture and log tokenization

    # Generate the Graphviz visualization
    dot = Digraph()
    dot.attr(rankdir="LR")  # Left to right layout for lexer tokens

    token_nodes = []
    for i, token in enumerate(tokens):
        token_name = f"Token_{i}"
        dot.node(token_name, f"{token.type}\n{token.value}")
        token_nodes.append(token_name)

        # Connect the tokens in sequence
        if i > 0:
            dot.edge(token_nodes[i - 1], token_name)

    # Render the tokenization visualization to a PNG file
    token_output_file = "tokenization_process"
    dot.render(token_output_file, format="png", cleanup=True)

    # Create a table for the Production and Semantic Analysis
    production_data = []
    for token in tokens:
        production_data.append([f"{token.type} -> {token.value}", f"Synthesized: {token.value}"])

    # Convert the production data to a DataFrame
    df = pd.DataFrame(production_data, columns=["Production", "Semantic Analysis"])

    # Plot the DataFrame as a table using Matplotlib
    fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the table
    ax.axis('tight')
    ax.axis('off')
    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    # Save the table as an image
    table_output_file = "production_table.png"
    plt.savefig(table_output_file, bbox_inches="tight")

    return token_output_file, table_output_file

# Main function
def main():
    expression = input("Enter a mathematical expression: ").strip()
    if not expression:
        print("Error: The expression is empty.")
        return

    if not is_balanced(expression):
        print("Error: Unbalanced parentheses in the expression.")
        return

    try:
        # Standalone evaluation
        result = evaluate_expression(expression)
        print(f"Standalone Evaluation: {result}")  # Labeling the standalone evaluation

        # Parse the input expression (for visualization and synthesized attributes)
        parse_tree = parser.parse(expression)

        # Compute synthesized attributes for the root node
        compute_attributes(parse_tree)

        # Visualize the parse tree
        dot = visualize_tree(parse_tree, expression)
        output_file = "math_expression_tree"
        dot.render(output_file, format="png", cleanup=True)

        # Generate tokenization visualization and production table
        tokenization_png, table_png = visualize_tokenization(expression)
        print(f"Tokenization process visualized in: {tokenization_png}.")
        print(f"Production table saved to: {table_png}.")
        print(f"Lexer operations logged to: lexer_operations_log.txt.")

    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
