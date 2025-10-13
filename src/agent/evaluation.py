# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Evaluation module for the Juno system."""

import time
import uuid
import json
from typing import Dict, Any, List, Tuple, Optional, Union
from dataclasses import asdict

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableConfig, RunnablePassthrough

from agent.state import State, TaskMetrics, ResourceConfig
from agent.configuration import Configuration
from agent.resource_monitor import calculate_efficiency_change


class JunoEvaluator:
    """Evaluator for Juno system performance and improvement tracking."""

    def __init__(self, config: Optional[Configuration] = None):
        """Initialize the evaluator.
        
        Args:
            config: Configuration parameters
        """
        self.config = config or Configuration()
        self.llm = init_chat_model(
            self.config.model_name,
            model_provider=self.config.model_provider,
        )
        self.eval_metrics = {}
        self.eval_id = str(uuid.uuid4())
    
    def evaluate_task_performance(
        self,
        state: State
    ) -> Dict[str, Any]:
        """Evaluate overall task performance metrics across multiple cycles.
        
        Args:
            state: The current system state
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get all task metrics
        metrics = state.get("metrics", [])
        
        if not metrics:
            return {
                "eval_id": self.eval_id,
                "timestamp": time.time(),
                "metrics": {},
                "summary": "Insufficient data to evaluate task performance."
            }
        
        # Calculate basic metrics
        total_tasks = len(metrics)
        success_rate = sum(1 for m in metrics if m.success) / total_tasks if total_tasks else 0
        avg_quality = sum(m.response_quality for m in metrics) / total_tasks if total_tasks else 0
        
        # Calculate deadline compliance
        deadline_met_count = sum(1 for m in metrics if getattr(m, "deadline_met", True))
        deadline_met_rate = deadline_met_count / total_tasks if total_tasks else 1.0
        
        # Task size/complexity metrics
        avg_task_size = sum(getattr(m, "task_size", 1.0) for m in metrics) / total_tasks if total_tasks else 1.0
        
        # Response time metrics
        avg_duration = sum(m.duration for m in metrics) / total_tasks if total_tasks else 0
        
        # Team-specific metrics
        teams = set(m.team_name for m in metrics if m.team_name)
        team_metrics = {}
        
        for team in teams:
            team_tasks = [m for m in metrics if m.team_name == team]
            team_count = len(team_tasks)
            if team_count > 0:
                team_success_rate = sum(1 for m in team_tasks if m.success) / team_count
                team_avg_quality = sum(m.response_quality for m in team_tasks) / team_count
                team_deadline_met_rate = sum(1 for m in team_tasks if getattr(m, "deadline_met", True)) / team_count
                
                team_metrics[team] = {
                    "task_count": team_count,
                    "success_rate": team_success_rate,
                    "avg_quality": team_avg_quality,
                    "deadline_met_rate": team_deadline_met_rate
                }
        
        # Compare with targets
        targets = state.get("performance_targets", [])
        target_achievement = {}
        
        for target in targets:
            metric_name = target.metric_name
            target_value = target.target_value
            current_value = 0.0
            
            # Map target metrics to calculated metrics
            if metric_name == "success_rate":
                current_value = success_rate
            elif metric_name == "response_quality" or metric_name == "avg_quality":
                current_value = avg_quality
            elif metric_name == "avg_response_time" or metric_name == "avg_duration":
                current_value = avg_duration
            elif metric_name == "deadline_met_rate":
                current_value = deadline_met_rate
            
            target_achievement[metric_name] = {
                "target": target_value,
                "current": current_value,
                "achieved": current_value >= target_value,
                "gap": target_value - current_value if current_value < target_value else 0
            }
        
        # Overall evaluation scores
        overall_score = (
            success_rate * 0.25 +
            avg_quality * 0.35 +
            deadline_met_rate * 0.4
        )
        
        # Save metrics for future comparison
        self.eval_metrics[self.eval_id] = {
            "timestamp": time.time(),
            "overall_score": overall_score,
            "success_rate": success_rate,
            "avg_quality": avg_quality,
            "deadline_met_rate": deadline_met_rate,
            "avg_task_size": avg_task_size
        }
        
        return {
            "eval_id": self.eval_id,
            "timestamp": time.time(),
            "metrics": {
                "total_tasks": total_tasks,
                "success_rate": success_rate,
                "avg_quality": avg_quality,
                "deadline_met_rate": deadline_met_rate,
                "avg_task_size": avg_task_size,
                "avg_duration": avg_duration,
                "overall_score": overall_score
            },
            "team_metrics": team_metrics,
            "target_achievement": target_achievement,
            "summary": f"Overall system performance score: {overall_score:.2f}/1.0"
        }
    
    def evaluate_code_improvements(
        self,
        state: State,
        baseline_eval_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """Evaluate the impact of code improvements compared to baseline.
        
        Args:
            state: The current system state
            baseline_eval_id: Optional ID of baseline evaluation for comparison
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get code changes
        code_changes = state.get("code_changes", {})
        fixes_implemented = state.get("fixes_implemented", [])
        
        if not code_changes:
            return {
                "eval_id": self.eval_id,
                "timestamp": time.time(),
                "improvement_impact": 0.0,
                "summary": "No code improvements have been implemented."
            }
        
        # Get current performance metrics
        current_eval = self.evaluate_task_performance(state)
        current_metrics = current_eval["metrics"]
        
        # Get baseline metrics for comparison
        baseline_metrics = None
        if baseline_eval_id and baseline_eval_id in self.eval_metrics:
            baseline_metrics = self.eval_metrics[baseline_eval_id]
        else:
            # Use the earliest evaluation as baseline
            sorted_evals = sorted(self.eval_metrics.items(), key=lambda x: x[1]["timestamp"])
            if sorted_evals:
                baseline_eval_id, baseline_metrics = sorted_evals[0]
        
        if not baseline_metrics:
            return {
                "eval_id": self.eval_id,
                "timestamp": time.time(),
                "fixes_implemented": len(fixes_implemented),
                "improvement_impact": 0.0,
                "current_metrics": current_metrics,
                "summary": "No baseline metrics available for comparison."
            }
        
        # Calculate improvement impact
        improvement_scores = {}
        
        for metric in ["overall_score", "success_rate", "avg_quality", "deadline_met_rate"]:
            if metric in current_metrics and metric in baseline_metrics:
                current_value = current_metrics[metric]
                baseline_value = baseline_metrics[metric]
                
                # Calculate relative improvement
                if baseline_value > 0:
                    relative_improvement = (current_value - baseline_value) / baseline_value
                else:
                    relative_improvement = current_value
                
                improvement_scores[metric] = {
                    "baseline": baseline_value,
                    "current": current_value,
                    "absolute_change": current_value - baseline_value,
                    "relative_change": relative_improvement
                }
        
        # Calculate task complexity adjustment
        complexity_factor = 1.0
        if "avg_task_size" in current_metrics and "avg_task_size" in baseline_metrics:
            current_complexity = current_metrics["avg_task_size"]
            baseline_complexity = baseline_metrics["avg_task_size"]
            
            if baseline_complexity > 0 and current_complexity > baseline_complexity:
                # Tasks got more complex, adjust improvement scores upward
                complexity_factor = current_complexity / baseline_complexity
        
        # Overall improvement score, adjusted for complexity
        overall_improvement = (
            improvement_scores.get("overall_score", {}).get("relative_change", 0) * complexity_factor
        )
        
        # Generate summary of improvements
        change_list = []
        for change_id, change_data in sorted(
            code_changes.items(), key=lambda x: x[1].get("timestamp", 0)
        ):
            issues_fixed = change_data.get("issues_fixed", [])
            if issues_fixed:
                change_list.append({
                    "change_id": change_id,
                    "timestamp": change_data.get("timestamp"),
                    "issues_fixed": issues_fixed
                })
        
        return {
            "eval_id": self.eval_id,
            "timestamp": time.time(),
            "baseline_eval_id": baseline_eval_id,
            "fixes_implemented": len(fixes_implemented),
            "improvement_scores": improvement_scores,
            "complexity_factor": complexity_factor,
            "overall_improvement": overall_improvement,
            "changes": change_list,
            "current_metrics": current_metrics,
            "baseline_metrics": baseline_metrics,
            "summary": f"Code improvements resulted in {overall_improvement:.1%} performance gain."
        }
    
    def evaluate_resource_scaling(
        self,
        state: State
    ) -> Dict[str, Any]:
        """Evaluate the impact of resource scaling.
        
        Args:
            state: The current system state
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Get resource change requests
        resource_requests = state.get("resource_change_requests", [])
        team_resources = state.get("team_resources", {})
        
        if not resource_requests:
            return {
                "eval_id": self.eval_id,
                "timestamp": time.time(),
                "scaling_effectiveness": 0.0,
                "summary": "No resource scaling has been performed."
            }
        
        # Group by team
        team_scaling = {}
        for team_name, resource_config in team_resources.items():
            team_requests = [r for r in resource_requests if r.get("team") == team_name]
            
            if team_requests:
                # Get the most recent request
                latest_request = max(team_requests, key=lambda x: x.get("timestamp", 0))
                
                # Calculate metrics before and after scaling
                metrics_before = [
                    m for m in state.get("metrics", [])
                    if m.team_name == team_name and m.start_time < latest_request.get("timestamp", 0)
                ]
                
                metrics_after = [
                    m for m in state.get("metrics", [])
                    if m.team_name == team_name and m.start_time >= latest_request.get("timestamp", 0)
                ]
                
                if metrics_before and metrics_after:
                    # Calculate performance metrics
                    before_performance = {
                        "avg_quality": sum(m.response_quality for m in metrics_before) / len(metrics_before),
                        "success_rate": sum(1 for m in metrics_before if m.success) / len(metrics_before),
                        "deadline_met_rate": sum(1 for m in metrics_before if getattr(m, "deadline_met", True)) / len(metrics_before)
                    }
                    
                    after_performance = {
                        "avg_quality": sum(m.response_quality for m in metrics_after) / len(metrics_after),
                        "success_rate": sum(1 for m in metrics_after if m.success) / len(metrics_after),
                        "deadline_met_rate": sum(1 for m in metrics_after if getattr(m, "deadline_met", True)) / len(metrics_after)
                    }
                    
                    # Calculate efficiency change
                    old_agent_count = latest_request.get("current_agents", 1)
                    new_agent_count = latest_request.get("recommended_agents", old_agent_count + 1)
                    
                    efficiency_change = calculate_efficiency_change(
                        before_performance,
                        after_performance,
                        old_agent_count,
                        new_agent_count
                    )
                    
                    # Store results
                    team_scaling[team_name] = {
                        "old_agents": old_agent_count,
                        "new_agents": new_agent_count,
                        "resource_increase": new_agent_count / old_agent_count,
                        "before_performance": before_performance,
                        "after_performance": after_performance,
                        "efficiency_change": efficiency_change
                    }
        
        # Calculate overall effectiveness
        if team_scaling:
            overall_effectiveness = sum(
                s["efficiency_change"] for s in team_scaling.values()
            ) / len(team_scaling)
        else:
            overall_effectiveness = 0.0
        
        return {
            "eval_id": self.eval_id,
            "timestamp": time.time(),
            "team_scaling": team_scaling,
            "overall_effectiveness": overall_effectiveness,
            "summary": f"Resource scaling effectiveness: {overall_effectiveness:.1%}"
        }
    
    def generate_evaluation_report(
        self,
        state: State, 
        config: RunnableConfig
    ) -> Dict[str, Any]:
        """Generate a comprehensive evaluation report.
        
        Args:
            state: The current system state
            config: Configuration parameters
            
        Returns:
            Dictionary with the evaluation report
        """
        # Run all evaluations
        performance_eval = self.evaluate_task_performance(state)
        improvement_eval = self.evaluate_code_improvements(state)
        scaling_eval = self.evaluate_resource_scaling(state)
        
        # Create a comprehensive report
        task_metrics = asdict(state.get("current_task", {})) if state.get("current_task") else {}
        deadline_compliance = state.get("missed_deadlines_count", 0)
        
        # Generate an LLM analysis
        prompt = ChatPromptTemplate.from_template(
            """You are an expert system evaluator analyzing the performance of an AI system called Juno.
            
            Please analyze the following evaluation data and provide insights on:
            1. Overall system performance
            2. Impact of code improvements
            3. Effectiveness of resource scaling
            4. Key areas for further improvement
            
            Task Performance Metrics:
            {performance_metrics}
            
            Code Improvement Impact:
            {improvement_metrics}
            
            Resource Scaling Effectiveness:
            {scaling_metrics}
            
            Deadline Compliance:
            Missed deadlines: {missed_deadlines}
            
            Provide a concise analysis with specific recommendations for further system optimization.
            Format your response as JSON with the following keys:
            - overall_assessment: Your overall assessment of the system's performance
            - strengths: List of system strengths
            - weaknesses: List of system weaknesses
            - improvement_recommendations: List of specific recommendations for further improvement
            - scaling_recommendations: Recommendations for optimal resource allocation
            """
        )
        
        # Format metrics for LLM
        performance_json = json.dumps(performance_eval["metrics"])
        improvement_json = json.dumps({
            "overall_improvement": improvement_eval["overall_improvement"],
            "fixes_implemented": improvement_eval["fixes_implemented"]
        })
        scaling_json = json.dumps({
            "overall_effectiveness": scaling_eval["overall_effectiveness"],
            "teams": list(scaling_eval.get("team_scaling", {}).keys())
        })
        
        # Run LLM analysis
        analysis_chain = (
            {
                "performance_metrics": lambda _: performance_json,
                "improvement_metrics": lambda _: improvement_json,
                "scaling_metrics": lambda _: scaling_json,
                "missed_deadlines": lambda _: deadline_compliance
            }
            | prompt
            | self.llm
            | StrOutputParser()
        )
        
        llm_analysis = analysis_chain.invoke({})
        
        try:
            # Parse JSON response
            analysis_json = json.loads(llm_analysis)
        except json.JSONDecodeError:
            # Fallback if not valid JSON
            analysis_json = {
                "overall_assessment": "Analysis error: Could not parse LLM output.",
                "improvement_recommendations": ["Review system logs for detailed metrics."]
            }
        
        # Build final report
        report = {
            "report_id": str(uuid.uuid4()),
            "timestamp": time.time(),
            "performance": performance_eval,
            "code_improvements": improvement_eval,
            "resource_scaling": scaling_eval,
            "analysis": analysis_json,
            "summary": analysis_json.get("overall_assessment", "No assessment available.")
        }
        
        return report