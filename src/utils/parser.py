import argparse
import ast


def parse_nested_list(input_string):
    """Convert a string representation of a nested list to an actual Python
    list."""
    try:
        # Use `ast.literal_eval` to safely evaluate the string as a Python
        # expression
        parsed_list = ast.literal_eval(input_string)
        if isinstance(parsed_list, list):
            return parsed_list
        else:
            raise ValueError
    except (ValueError, SyntaxError):
        raise argparse.ArgumentTypeError(f"Invalid list format: {input_string}")
