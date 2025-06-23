from abc import ABC, abstractmethod
import torch
from sympy import *
import re
import latex2sympy
import numpy as np
from typing import Optional, Tuple

from src.models.llada.loss.loss_utils import LossUtils


class LossMetric(ABC):
    @abstractmethod
    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Returns a Tensor between 0 and self.weight representing the reward value for the model output. 
        Must be implemented by subclasses.
        """
        pass


class AntiGamingMetric(LossMetric):
    def __init__(self, weight: Tuple[float, float] = (-5.0, 0.0)):
        """
        Metric to detect and penalize gaming behaviors like generic placeholders.
        
        Args:
            weight: Penalty weight for detected gaming behaviors
        """
        self.weight = weight
        
        self.gaming_patterns = [
            r'\\boxed\{\(solution\)\}',  # Literal "(solution)"
            r'\\boxed\{solution\}',      # Literal "solution"
            r'\\boxed\{answer\}',        # Literal "answer"
            r'\\boxed\{result\}',        # Literal "result"
            r'\\boxed\{\.\.\.\}',        # Ellipsis
            r'\\boxed\{placeholder\}',   # Placeholder text
            r'\\boxed\{ANSWER\}',        # All caps variations
            r'\\boxed\{SOLUTION\}',
            r'\\boxed\{RESULT\}',
        ]
        
        self.placeholder_patterns = [
            r'^solution$',
            r'^\(solution\)$',
            r'^answer$',
            r'^result$',
            r'^placeholder$',
            r'^todo$',
            r'^\.\.\.$',
            r'^…$',
        ]
    
    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Detect gaming behaviors in model output.
        
        Args:
            model_output: The model's output as a string
            ground_truth: The target data (not used for anti-gaming)
            
        Returns:
            torch.Tensor: Penalty for detected gaming behaviors
        """
        reward = torch.tensor(0.0)
        
        if isinstance(model_output, list):
            model_output = model_output[0]
        
        output_lower = model_output.lower().strip()
        
        for pattern in self.gaming_patterns:
            if re.search(pattern, output_lower, re.IGNORECASE):
                reward += self.weight[0] 
                return reward
        
        boxed_content = LossUtils.extract_boxed_content(model_output)
        if boxed_content:
            boxed_lower = boxed_content.lower().strip()
            for pattern in self.placeholder_patterns:
                if re.match(pattern, boxed_lower, re.IGNORECASE):
                    reward += self.weight[0] 
                    return reward
        
        if len(output_lower.replace(" ", "")) < 3:
            reward += self.weight[0] 
            return reward
        
        if re.match(r'^[^\w]*$', output_lower):
            reward += self.weight[0] 
            return reward
        
        return reward


class MathSimilarityMetric(LossMetric):
    def __init__(self, use_boxed: bool = True, weight: Tuple[float, float] = (-1.0, 1.0), 
                 similarity_threshold: float = 0.7, partial_credit: bool = True):
        """
        AST-based mathematical similarity metric for partial credit.
        
        Args:
            use_boxed: Whether to extract content from \\boxed{}
            weight: Weight for this metric  
            similarity_threshold: Minimum similarity for partial credit
            partial_credit: Whether to give partial credit for similar structures
        """
        self.use_boxed = use_boxed
        self.weight = weight
        self.similarity_threshold = similarity_threshold
        self.partial_credit = partial_credit
    
    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Calculate AST-based similarity between model output and ground truth.
        
        Args:
            model_output: The model's output as a string
            ground_truth: The target data for the problem
            
        Returns:
            torch.Tensor: Similarity-based reward
        """
        reward = torch.tensor(0.0)
        
        if isinstance(model_output, list):
            model_output = model_output[0]
        
        # Get ground truth solution
        if 'solution' not in ground_truth or not ground_truth['solution']:
            if "variant" not in ground_truth or not ground_truth['variant']:
                return reward
            variant = ground_truth['variant'].lower()
            try:
                x = symbols('x')
                if isinstance(variant, str):
                    if variant.startswith('integrate('):
                        match = re.search(r'integrate\((.+),\s*x\)', variant)
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
        
        # Extract model prediction
        if self.use_boxed:
            model_output = LossUtils.extract_boxed_content(model_output)
            if model_output == "":
                reward += self.weight[0]
                return reward
        
        model_sympy = LossUtils.get_prediction_from_model_output(model_output)
        if not model_sympy:
            reward += self.weight[0]
            return reward
        
        # Check exact equality first
        if LossUtils.check_sympy_equality(model_sympy, solution):
            reward += self.weight[1]
            return reward
        
        if not self.partial_credit:
            reward += self.weight[0]
            return reward
        
        # Calculate AST similarity for partial credit
        try:
            similarity = self._calculate_ast_similarity(model_sympy, solution)
            
            if similarity >= self.similarity_threshold:
                # Interpolate reward based on similarity
                partial_reward = similarity * self.weight[1]
                reward += partial_reward
            else:
                reward += self.weight[0]
                
        except Exception as e:
            print(f"Error calculating AST similarity: {e}")
            reward += self.weight[0]
        
        return reward
    
    def _calculate_ast_similarity(self, expr1_str: str, expr2_str: str) -> float:
        """
        Calculate structural similarity between two mathematical expressions using AST.
        
        Args:
            expr1_str: First expression as string
            expr2_str: Second expression as string
            
        Returns:
            float: Similarity score between 0 and 1
        """
        try:
            expr1 = sympify(expr1_str)
            expr2 = sympify(expr2_str)
            
            # Get AST representations
            ast1 = self._get_expression_signature(expr1)
            ast2 = self._get_expression_signature(expr2)
            
            # Calculate structural similarity
            similarity = self._compare_signatures(ast1, ast2)
            
            return similarity
            
        except Exception as e:
            print(f"Error in AST similarity calculation: {e}")
            return 0.0
    
    def _get_expression_signature(self, expr) -> dict:
        """
        Extract structural signature from SymPy expression.
        
        Args:
            expr: SymPy expression
            
        Returns:
            dict: Structural signature of the expression
        """
        if expr.is_Atom:
            if expr.is_Symbol:
                return {"type": "Symbol", "count": 1}
            elif expr.is_Number:
                return {"type": "Number", "count": 1}
            else:
                return {"type": "Atom", "count": 1}
        
        signature = {
            "type": type(expr).__name__,
            "count": 1,
            "children": []
        }
        
        for arg in expr.args:
            child_sig = self._get_expression_signature(arg)
            signature["children"].append(child_sig)
        
        return signature
    
    def _compare_signatures(self, sig1: dict, sig2: dict) -> float:
        """
        Compare two expression signatures for similarity.
        
        Args:
            sig1: First signature
            sig2: Second signature
            
        Returns:
            float: Similarity score between 0 and 1
        """
        # Type similarity
        type_match = 1.0 if sig1["type"] == sig2["type"] else 0.0
        
        # If no children, return type similarity
        if not sig1.get("children") and not sig2.get("children"):
            return type_match
        
        # If one has children and other doesn't, partial similarity
        if not sig1.get("children") or not sig2.get("children"):
            return type_match * 0.5
        
        # Compare children structures
        children1 = sig1["children"]
        children2 = sig2["children"]
        
        # Create similarity matrix between children
        n1, n2 = len(children1), len(children2)
        if n1 == 0 and n2 == 0:
            children_similarity = 1.0
        elif n1 == 0 or n2 == 0:
            children_similarity = 0.0
        else:
            # Find best matching between children
            similarities = []
            for c1 in children1:
                max_sim = 0.0
                for c2 in children2:
                    sim = self._compare_signatures(c1, c2)
                    max_sim = max(max_sim, sim)
                similarities.append(max_sim)
            
            children_similarity = sum(similarities) / max(n1, n2)
        
        # weight type similarity and children similarity
        total_similarity = 0.4 * type_match + 0.6 * children_similarity
        
        return total_similarity


class LengthRewardMetric(LossMetric):
    def __init__(self, min_tokens: int = 10, weight: Tuple[float, float] = (-1.0, 1.0)):
        """
        Reward metric that encourages responses with adequate length.
        
        Args:
            min_tokens: Minimum number of tokens to receive full reward
            weight: Weight for this metric
        """
        self.min_tokens = min_tokens
        self.weight = weight
    
    def __call__(self, model_output: str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Get the reward for response length.
        
        Args:
            model_output: The model's output as a string
            ground_truth: The target data (not used for length metric)
            
        Returns:
            torch.Tensor: The reward for response length
        """
        reward = torch.tensor(0.0)
        
        if isinstance(model_output, list):
            model_output = model_output[0]
        
        output_length = len(model_output)
        
        if output_length >= self.min_tokens:
            reward += self.weight[1]
        else:
            if output_length >= 10:
                reward += 1 * self.weight[1]
            elif output_length >= 5:
                reward += 0.5 * self.weight[1]
            else:
                reward += 1.0 * self.weight[0]
                
        return reward


class MathCorrectnessMetric(LossMetric):
    def __init__(self, use_boxed: bool = False, regex_match: Optional[str] = None, weight: Tuple[float, float] = (-1.0, 1.0)):
        assert sum((int(regex_match is not None), int(use_boxed))) == 1, "Only one of regex_match or use_boxed must be provided."
        self.use_boxed = use_boxed
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
        if self.use_boxed:
            model_output = LossUtils.extract_boxed_content(model_output)
            if model_output == "":
                reward += self.weight[0]
                return reward
        else:
            model_output = re.search(self.regex_match, model_output, re.DOTALL)
            if model_output is None:
                reward += self.weight[0]
                return reward
            else:
                model_output = model_output.group(1)
        
        model_sympy = LossUtils.get_prediction_from_model_output(model_output)
        if not model_sympy:
            reward += self.weight[0]
            return reward

        # Use SymPy equality checking for more robust comparison
        if LossUtils.check_sympy_equality(model_sympy, solution) or model_sympy == solution or model_output == solution_latex:
            reward += self.weight[1]
            
        return reward 
    


class MathFormatMetric(LossMetric):
    def __init__(self, use_boxed: bool = False, regex_match: Optional[str] = None, weight: Tuple[float, float] = (-1.0, 1.0)):
        assert sum((int(regex_match is not None), int(use_boxed))) == 1, "Only one of regex_match or use_boxed must be provided."
        self.use_boxed = use_boxed
        self.regex_match = regex_match
        self.weight = weight

    def __call__(self, model_output: list[str] | str, ground_truth: dict, **kwargs) -> torch.Tensor:
        """
        Get the reward for the math format.
        """
        reward = torch.tensor(0.0)

        if isinstance(model_output, list):
            model_output = model_output[0]

        if self.use_boxed:
            model_output = LossUtils.extract_boxed_content(model_output)
            if model_output == "":
                reward += self.weight[0]
                return reward
        else:
            model_output = re.search(self.regex_match, model_output, re.DOTALL)
            if model_output is None:
                reward += self.weight[0]
                return reward
            else:
                model_output = model_output.group(1)

        match = bool(model_output)
        if match:
            reward += self.weight[1]
        else:
            reward += self.weight[0]
        return reward
    

class MathEvalMetric(LossMetric):
    def __init__(self, use_boxed: bool = False, regex_match: Optional[str] = None, weight: Tuple[float, float] = (-1.0, 1.0), num_test_points: int = 100, tolerance: float = 1e-4):
        assert sum((int(regex_match is not None), int(use_boxed))) == 1, "Only one of regex_match or use_boxed must be provided."
        self.use_boxed = use_boxed
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
        print(model_output)
        if self.use_boxed:
            model_output = LossUtils.extract_boxed_content(model_output)
            print(f"Model output: {model_output}")
            if model_output == "":
                reward += self.weight[0]
                return reward
        else:
            model_output = re.search(self.regex_match, model_output, re.DOTALL)
            if model_output is None:
                reward += self.weight[0]
                return reward
            else:
                model_output = model_output.group(1)
        
        model_sympy = LossUtils.get_prediction_from_model_output(model_output)
        print(f"Model sympy: {model_sympy}")
        print(f"Solution: {solution}")
        if not model_sympy:
            reward += self.weight[0]
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
                reward += self.weight[1]
            elif success_ratio >= 0.7:  # partial reward for 70-89% match
                reward += 0.5 * self.weight[1]
            else:
                reward += 1.0 * self.weight[0]  # otherwise penalize incorrect answers
                
        except Exception as e:
            print(f"Error in numerical evaluation: {e}")
            return reward

        return reward
    