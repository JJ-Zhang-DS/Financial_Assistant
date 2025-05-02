import numpy as np
from typing import Dict, List, Any, Tuple
import json
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import ParameterGrid

class FinancialRAGTuner:
    """Parameter tuner for financial RAG systems."""
    
    def __init__(self, agent, evaluation_function, test_dataset):
        """Initialize the tuner with agent and evaluation method."""
        self.agent = agent
        self.evaluate = evaluation_function
        self.test_dataset = test_dataset
        self.results = []
        self.best_params = {}
        self.best_score = 0
    
    def tune_retrieval_parameters(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tune retrieval parameters for optimal performance."""
        print("Tuning retrieval parameters...")
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        best_score = 0
        best_params = {}
        
        # Test each parameter combination
        for params in tqdm(param_combinations):
            # Apply parameters to agent
            self._apply_retrieval_params(params)
            
            # Evaluate performance
            eval_results = self.evaluate(self.agent, self.test_dataset)
            score = eval_results.get("overall_score", 0)
            
            # Record results
            result = {
                "params": params,
                "score": score,
                "eval_details": eval_results
            }
            self.results.append(result)
            
            # Update best parameters if improved
            if score > best_score:
                best_score = score
                best_params = params
        
        # Apply best parameters
        self._apply_retrieval_params(best_params)
        self.best_params["retrieval"] = best_params
        
        print(f"Best retrieval parameters: {best_params}")
        print(f"Best score: {best_score}")
        
        return best_params
    
    def tune_generation_parameters(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tune generation parameters for optimal performance."""
        print("Tuning generation parameters...")
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        best_score = 0
        best_params = {}
        
        # Test each parameter combination
        for params in tqdm(param_combinations):
            # Apply parameters to agent
            self._apply_generation_params(params)
            
            # Evaluate performance
            eval_results = self.evaluate(self.agent, self.test_dataset)
            score = eval_results.get("overall_score", 0)
            
            # Record results
            result = {
                "params": params,
                "score": score,
                "eval_details": eval_results
            }
            self.results.append(result)
            
            # Update best parameters if improved
            if score > best_score:
                best_score = score
                best_params = params
        
        # Apply best parameters
        self._apply_generation_params(best_params)
        self.best_params["generation"] = best_params
        
        print(f"Best generation parameters: {best_params}")
        print(f"Best score: {best_score}")
        
        return best_params
    
    def tune_context_assembly_parameters(self, param_grid: Dict[str, List[Any]]) -> Dict[str, Any]:
        """Tune context assembly parameters for optimal performance."""
        print("Tuning context assembly parameters...")
        
        # Generate all parameter combinations
        param_combinations = list(ParameterGrid(param_grid))
        best_score = 0
        best_params = {}
        
        # Test each parameter combination
        for params in tqdm(param_combinations):
            # Apply parameters to agent
            self._apply_context_params(params)
            
            # Evaluate performance
            eval_results = self.evaluate(self.agent, self.test_dataset)
            score = eval_results.get("overall_score", 0)
            
            # Record results
            result = {
                "params": params,
                "score": score,
                "eval_details": eval_results
            }
            self.results.append(result)
            
            # Update best parameters if improved
            if score > best_score:
                best_score = score
                best_params = params
        
        # Apply best parameters
        self._apply_context_params(best_params)
        self.best_params["context_assembly"] = best_params
        
        print(f"Best context assembly parameters: {best_params}")
        print(f"Best score: {best_score}")
        
        return best_params
    
    def _apply_retrieval_params(self, params: Dict[str, Any]) -> None:
        """Apply retrieval parameters to the agent."""
        # This implementation depends on your agent's interface
        # Example implementation:
        if hasattr(self.agent, "retriever_config"):
            for key, value in params.items():
                if key == "k":
                    self.agent.retriever_config["search_kwargs"]["k"] = value
                elif key == "fetch_k":
                    self.agent.retriever_config["search_kwargs"]["fetch_k"] = value
                elif key == "lambda_mult":
                    self.agent.retriever_config["search_kwargs"]["lambda_mult"] = value
                elif key == "score_threshold":
                    self.agent.retriever_config["search_kwargs"]["score_threshold"] = value
            
            # Reinitialize retriever with new config
            if hasattr(self.agent, "initialize_retriever"):
                self.agent.initialize_retriever()
    
    def _apply_generation_params(self, params: Dict[str, Any]) -> None:
        """Apply generation parameters to the agent."""
        # This implementation depends on your agent's interface
        # Example implementation:
        if hasattr(self.agent, "llm"):
            for key, value in params.items():
                if key == "temperature":
                    self.agent.llm.temperature = value
                elif key == "model_name":
                    # Create new LLM with different model
                    from langchain.chat_models import ChatOpenAI
                    self.agent.llm = ChatOpenAI(
                        model_name=value,
                        temperature=self.agent.llm.temperature
                    )
    
    def _apply_context_params(self, params: Dict[str, Any]) -> None:
        """Apply context assembly parameters to the agent."""
        # This implementation depends on your agent's interface
        # Example implementation:
        if hasattr(self.agent, "context_config"):
            for key, value in params.items():
                self.agent.context_config[key] = value
    
    def visualize_parameter_impact(self, param_name: str) -> None:
        """Visualize the impact of a specific parameter on performance."""
        # Extract results for the parameter
        param_values = []
        scores = []
        
        for result in self.results:
            if param_name in result["params"]:
                param_values.append(result["params"][param_name])
                scores.append(result["score"])
        
        if not param_values:
            print(f"No data available for parameter: {param_name}")
            return
        
        # Create visualization
        plt.figure(figsize=(10, 6))
        
        # Check if parameter is numeric
        if all(isinstance(x, (int, float)) for x in param_values):
            # For numeric parameters, use line plot
            df = pd.DataFrame({
                "Parameter Value": param_values,
                "Score": scores
            })
            
            # Group by parameter value and take mean
            grouped = df.groupby("Parameter Value").mean().reset_index()
            
            # Sort by parameter value
            grouped = grouped.sort_values("Parameter Value")
            
            # Plot
            plt.plot(grouped["Parameter Value"], grouped["Score"], marker='o')
            plt.xlabel(f"Parameter Value: {param_name}")
            plt.ylabel("Performance Score")
            
            # Add trend line
            sns.regplot(x="Parameter Value", y="Score", data=grouped, scatter=False, ci=None)
        else:
            # For categorical parameters, use bar plot
            df = pd.DataFrame({
                "Parameter Value": param_values,
                "Score": scores
            })
            
            # Group by parameter value and take mean
            grouped = df.groupby("Parameter Value").mean().reset_index()
            
            # Plot
            sns.barplot(x="Parameter Value", y="Score", data=grouped)
            plt.xlabel(f"Parameter Value: {param_name}")
            plt.ylabel("Performance Score")
        
        plt.title(f"Impact of {param_name} on Performance")
        plt.tight_layout()
        plt.savefig(f"parameter_impact_{param_name}.png")
        plt.close()
    
    def get_recommended_parameters(self) -> Dict[str, Dict[str, Any]]:
        """Get the recommended parameters for all components."""
        return self.best_params
    
    def generate_tuning_report(self) -> str:
        """Generate a report of the tuning process and results."""
        report = "# Financial RAG Parameter Tuning Report\n\n"
        
        # Add best parameters
        report += "## Recommended Parameters\n\n"
        
        if "retrieval" in self.best_params:
            report += "### Retrieval Parameters\n\n"
            report += "| Parameter | Value |\n|-----------|-------|\n"
            for key, value in self.best_params["retrieval"].items():
                report += f"| {key} | {value} |\n"
            report += "\n"
        
        if "generation" in self.best_params:
            report += "### Generation Parameters\n\n"
            report += "| Parameter | Value |\n|-----------|-------|\n"
            for key, value in self.best_params["generation"].items():
                report += f"| {key} | {value} |\n"
            report += "\n"
        
        if "context_assembly" in self.best_params:
            report += "### Context Assembly Parameters\n\n"
            report += "| Parameter | Value |\n|-----------|-------|\n"
            for key, value in self.best_params["context_assembly"].items():
                report += f"| {key} | {value} |\n"
            report += "\n"
        
        # Add parameter impact analysis
        report += "## Parameter Impact Analysis\n\n"
        
        # Get all unique parameters
        all_params = set()
        for result in self.results:
            all_params.update(result["params"].keys())
        
        # Generate visualizations for each parameter
        for param in all_params:
            self.visualize_parameter_impact(param)
            report += f"### Impact of {param}\n\n"
            report += f"![Impact of {param}](parameter_impact_{param}.png)\n\n"
        
        return report


# Example usage
def optimize_financial_rag_parameters():
    """Run parameter optimization for financial RAG."""
    from agents.finance_agent import FinanceAgent
    from evaluation.rag_evaluator import FinancialRAGEvaluator
    
    # Initialize agent
    agent = FinanceAgent()
    
    # Create evaluation function
    def evaluate(agent, test_dataset):
        evaluator = FinancialRAGEvaluator(agent)
        return evaluator.evaluate(test_dataset)
    
    # Create test dataset
    test_dataset = [
        # Your test cases here
    ]
    
    # Initialize tuner
    tuner = FinancialRAGTuner(agent, evaluate, test_dataset)
    
    # Tune retrieval parameters
    retrieval_param_grid = {
        "k": [3, 4, 5, 6],
        "fetch_k": [10, 15, 20],
        "lambda_mult": [0.5, 0.7, 0.9],
        "score_threshold": [0.65, 0.7, 0.75, 0.8]
    }
    tuner.tune_retrieval_parameters(retrieval_param_grid)
    
    # Tune generation parameters
    generation_param_grid = {
        "temperature": [0.0, 0.1, 0.2],
        "model_name": ["gpt-3.5-turbo", "gpt-4"]
    }
    tuner.tune_generation_parameters(generation_param_grid)
    
    # Tune context assembly parameters
    context_param_grid = {
        "max_tokens": [2000, 3000, 4000],
        "include_metadata": [True, False],
        "format_type": ["detailed", "concise"]
    }
    tuner.tune_context_assembly_parameters(context_param_grid)
    
    # Generate report
    report = tuner.generate_tuning_report()
    
    # Save report
    with open("financial_rag_tuning_report.md", "w") as f:
        f.write(report)
    
    return tuner.get_recommended_parameters() 