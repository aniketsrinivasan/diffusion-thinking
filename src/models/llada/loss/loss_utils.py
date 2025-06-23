from sympy import *
import re
from latex2sympy.latex2sympy import process_sympy
import torch


class LossUtils:
    @staticmethod
    def sanitize_latex(latex: str) -> str:
        """
        Sanitize the latex string.
        """
        replacements = {
            " ": "",
            "\n": "",
            "+c": "",
            "\\left": "",
            "\\right": "",
            "\\,": "",
            "\\.": "",
            "tan^{-1}": "arctan",
        }
        latex = latex.lower()
        for k, v in replacements.items():
            latex = latex.replace(k, v)
        if "=" in latex:
            latex = latex.split("=")[1]
        return latex
    
    @staticmethod
    def latex_to_sympy(latex: str) -> str:
        """
        Convert latex to sympy expression.
        """
        latex = LossUtils.sanitize_latex(latex)
        return str(process_sympy(latex))

    @staticmethod
    def get_prediction_from_model_output(model_output: str) -> str:
        """
        Get the prediction from the model output.
        """
        model_output = LossUtils.sanitize_latex(model_output)
        orig_output = model_output
        try:
            model_output = process_sympy(model_output)  # convert latex to sympy expression
            model_output = str(model_output)
        except Exception as e:
            print(f"Error converting latex to sympy: {e}")
            model_output = orig_output
        return model_output
    
    @staticmethod
    def check_sympy_equality(model_output: str, ground_truth: str) -> bool:
        """
        Check whether two SymPy expressions are equivalent.
        
        Args:
            model_output: String representation of the model's mathematical expression
            ground_truth: String representation of the ground truth expression
            
        Returns:
            bool: True if expressions are mathematically equivalent, False otherwise
        """
        try:
            x = symbols('x')
            
            expr1 = sympify(model_output)
            expr2 = sympify(ground_truth)
        
            return expr1.equals(expr2)
            
        except Exception as e:
            print(f"Error in sympy equality check: {e}")
            return model_output.strip() == ground_truth.strip()
    
    @staticmethod
    def extract_boxed_content(text: str) -> str:
        """
        Extract the LaTeX content inside \\boxed{} from a string.
        Handles nested braces correctly.
        
        Args:
            text: Input string that may contain \\boxed{...} pattern
            
        Returns:
            str: The content inside \\boxed{}, or empty string if not found
        """
        boxed_start = text.find(r'\boxed{')
        if boxed_start == -1:
            return ""
        
        start_pos = boxed_start + len(r'\boxed{')
        brace_count = 1
        current_pos = start_pos
        
        while current_pos < len(text) and brace_count > 0:
            char = text[current_pos]
            if char == '{':
                brace_count += 1
            elif char == '}':
                brace_count -= 1
            current_pos += 1
        
        if brace_count == 0:
            return text[start_pos:current_pos-1]
        
        return ""
