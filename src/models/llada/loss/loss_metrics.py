from abc import ABC, abstractmethod
import torch
from sympy import *
import re
import latex2sympy
import numpy as np

from src.models.llada.loss.loss_utils import LossUtils


class LossMetric(ABC):
    @abstractmethod
    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Returns a Tensor between 0 and self.weight representing the reward value for the model output. 
        Must be implemented by subclasses.
        """
        pass


class MathCorrectnessMetric(LossMetric):
    def __init__(self, regex_match: str, weight: float = 1.0):
        self.regex_match = regex_match
        self.weight = weight

    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Get the reward for the math correctness. This function calculates whether a model's mathematical
        expression matches the ground truth solution.

        If the ground truth is not found, the function attempts to evaluate the variant as a sympy expression.

        Args:
            model_output (str): The model's output as a string.
            ground_truth (dict): The target data for the problem. Typically must contain one of the following keys:
                - "solution": The solution to the problem.
                - "variant": The variant of the problem.

        Returns:
            torch.Tensor: The reward for the math correctness.
        """
        reward = torch.tensor(0.0)

        if isinstance(model_output, list):
            model_output = model_output[0]

        # get the solution to the problem: 
        if 'solution' not in ground_truth or not ground_truth['solution']:
            # evaluate the ground truth question:
            if "variant" not in ground_truth or not ground_truth['variant']:
                return reward 
            variant = ground_truth['variant'].lower()
            try:
                x = symbols('x')
                if isinstance(variant, str):
                    if variant.startswith('integrate('):
                        import re as regex_module  # avoid namespace conflict
                        match = regex_module.search(r'integrate\((.+),\s*x\)', variant)
                        if match:
                            expr_str = match.group(1)
                            expr = sympify(expr_str)
                        else:
                            expr = sympify(variant)
                    else:
                        expr = sympify(variant)
                else:
                    expr = variant
                solution = integrate(expr, x)
                solution = str(solution)
                
            except Exception as e:
                return reward
        else:
            solution = ground_truth['solution']
        
        solution = str(solution).lower().strip().replace(" ", "")
        try:
            solution_latex = str(latex2sympy.latex(solution.replace(" ", "").lower())).replace(" ", "").lower()
        except Exception as e:
            solution_latex = ""
        
        # get stuff inside substring:
        model_output = re.search(self.regex_match, model_output, re.DOTALL)

        if model_output is None:
            return reward
        else:
            model_output = model_output.group(1)
        
        model_sympy = LossUtils.get_prediction_from_model_output(model_output)
        if not model_sympy:
            return reward

        # Use SymPy equality checking for more robust comparison
        if LossUtils.check_sympy_equality(model_sympy, solution) or model_sympy == solution or model_output == solution_latex:
            reward += self.weight
            
        return reward 
    


class MathFormatMetric(LossMetric):
    def __init__(self, regex_match: str, weight: float = 1.0):
        self.regex_match = regex_match
        self.weight = weight

    def __call__(self, model_output: list[str] | str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Get the reward for the math format.
        """
        reward = torch.tensor(0.0)

        if isinstance(model_output, list):
            model_output = model_output[0]

        match = bool(re.search(self.regex_match, model_output, re.DOTALL))
        if match:
            reward += self.weight
        return reward
    

class MathEvalMetric(LossMetric):
    def __init__(self, regex_match: str, weight: float = 1.0, num_test_points: int = 10, tolerance: float = 1e-4):
        self.regex_match = regex_match
        self.weight = weight
        self.num_test_points = num_test_points
        self.tolerance = tolerance

    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Get the reward for the math evaluation.
        """
        reward = torch.tensor(0.0)

        if isinstance(model_output, list):
            model_output = model_output[0]

        # get the solution to the problem: 
        if 'solution' not in ground_truth or not ground_truth['solution']:
            # evaluate the ground truth question:
            if "variant" not in ground_truth or not ground_truth['variant']:
                return reward 
            variant = ground_truth['variant'].lower()
            try:
                x = symbols('x')
                if isinstance(variant, str):
                    if variant.startswith('integrate('):
                        import re as regex_module  # avoid namespace conflict
                        match = regex_module.search(r'integrate\((.+),\s*x\)', variant)
                        if match:
                            expr_str = match.group(1)
                            expr = sympify(expr_str)
                        else:
                            expr = sympify(variant)
                    else:
                        expr = sympify(variant)
                else:
                    expr = variant
                solution = integrate(expr, x)
                solution = str(solution)
                
            except Exception as e:
                return reward
        else:
            solution = ground_truth['solution']
        
        solution = str(solution).lower().strip().replace(" ", "")

        # get stuff inside substring:
        model_output = re.search(self.regex_match, model_output, re.DOTALL)

        if model_output is None:
            return reward
        else:
            model_output = model_output.group(1)
        
        model_sympy = LossUtils.get_prediction_from_model_output(model_output)
        if not model_sympy:
            return reward
        
        # numerical evaluation at multiple random x values sampled from normal(0, 1)
        try:
            x_symbol = symbols('x')
            model_expr = sympify(model_sympy)
            solution_expr = sympify(solution)
            
            test_x_values = np.random.normal(0, 1, self.num_test_points)
            
            matches = 0
            valid_evaluations = 0
            
            for x_val in test_x_values:
                try:
                    model_num = float(model_expr.subs(x_symbol, x_val))
                    solution_num = float(solution_expr.subs(x_symbol, x_val))
                    
                    valid_evaluations += 1
                    
                    if abs(model_num - solution_num) < self.tolerance:
                        matches += 1
                        
                except (ValueError, TypeError, AttributeError) as e:
                    continue
            
            if valid_evaluations == 0:
                return reward
                
            # calculate success ratio
            success_ratio = matches / valid_evaluations
            print(f"Numerical evaluation: {matches}/{valid_evaluations} matches ({success_ratio:.2%})")
            
            # reward based on success ratio (need high success rate for full reward)
            if success_ratio >= 0.9:  # 90% of points must match
                reward += self.weight
            elif success_ratio >= 0.7:  # partial reward for 70-89% match
                reward += 0.5 * self.weight
            else:
                reward -= 1.0 * self.weight  # otherwise penalize incorrect answers
                
        except Exception as e:
            print(f"Error in numerical evaluation: {e}")
            return reward

        return reward
    