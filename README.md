# Math Expression Parser

## Installation

To use the Math Expression Parser, you'll need to have the following dependencies installed:

- `lark`
- `graphviz`
- `pandas`
- `matplotlib`

You can install these dependencies using pip:

```
pip install lark-parser graphviz pandas matplotlib
```

## Usage

To use the Math Expression Parser, run the `main()` function in the `MathParser.py` file. The program will prompt you to enter a mathematical expression, and it will then perform the following actions:

1. Evaluate the expression using the BODMAS (Brackets, Orders, Division, Multiplication, Addition, Subtraction) rules.
2. Visualize the parse tree of the expression using Graphviz.
3. Visualize the tokenization process and generate a production table.

The output files will be saved in the same directory as the `MathParser.py` file.

## API

The `MathParser.py` file provides the following functions:

- `compute_attributes(tree, inherited_value=None)`: Computes the synthesized and inherited attributes for the nodes in the parse tree.
- `visualize_tree(parse_tree, raw_expression)`: Creates a visualization of the parse tree using Graphviz.
- `is_balanced(expression)`: Checks if the input expression has balanced parentheses.
- `evaluate_expression(expression)`: Evaluates the input expression using BODMAS rules.
- `lexer_callback(token, log_file)`: Captures and logs the tokenization process.
- `visualize_tokenization(expression)`: Visualizes the tokenization process and generates a production table.
- `main()`: The main function that prompts the user for an expression and performs the various operations.

## Contributing

If you find any issues or have suggestions for improvements, please feel free to open an issue or submit a pull request on the project's GitHub repository.

## License

This project is licensed under the [MIT License](LICENSE).

## Testing

To run the tests for the Math Expression Parser, you can use the built-in Python testing framework. Create a new file called `test_MathParser.py` in the same directory as `MathParser.py` and add your test cases.

You can then run the tests using the following command:

```
python -m unittest test_MathParser
```

Make sure to include comprehensive test cases to ensure the correct functionality of the Math Expression Parser.
