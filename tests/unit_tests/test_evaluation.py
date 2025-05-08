"""Unit tests for the evaluation system."""

import time
import json
import unittest
from unittest.mock import patch, MagicMock, ANY

from agent.state import State, TaskMetrics, PerformanceTarget
from agent.configuration import Configuration
from agent.evaluation import JunoEvaluator


class TestJunoEvaluator(unittest.TestCase):
    """Test cases for the JunoEvaluator class."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration()
        self.evaluator = JunoEvaluator(self.config)
        
        # Create test state
        self.state = State()
        
        # Add test metrics
        self.state.metrics = [
            TaskMetrics(
                task_id=f"task_{i}",
                team_name="research" if i % 2 == 0 else "writing",
                response_quality=0.7 + (i * 0.02),
                success=True,
                start_time=time.time() - 3600,
                end_time=time.time() - 3550,
                duration=50.0,
                deadline=time.time() - 3500,
                deadline_met=True,
                task_size=1.0 + (i * 0.1)
            ) for i in range(10)
        ]
        
        # Add performance targets
        self.state.performance_targets = [
            PerformanceTarget(metric_name="success_rate", target_value=0.9),
            PerformanceTarget(metric_name="avg_quality", target_value=0.8)
        ]
        
        # Add code changes
        self.state.fixes_implemented = ["Fixed bug in task assignment", "Improved error handling"]
        self.state.code_changes = {
            "change_1": {
                "issues_fixed": ["Low quality in research team", "Missed deadlines"],
                "implemented_fixes": ["Fixed bug in task assignment"],
                "timestamp": time.time() - 7200  # 2 hours ago
            },
            "change_2": {
                "issues_fixed": ["Error handling issues"],
                "implemented_fixes": ["Improved error handling"],
                "timestamp": time.time() - 3600  # 1 hour ago
            }
        }
        
        # Add resource changes
        self.state.resource_change_requests = [
            {
                "team": "research",
                "current_agents": 1,
                "recommended_agents": 2,
                "reason": "High deadline miss rate",
                "timestamp": time.time() - 3600
            }
        ]
        
        self.state.team_resources = {
            "research": MagicMock(current_agents=2, max_agents=3),
            "writing": MagicMock(current_agents=1, max_agents=3)
        }

    def test_evaluate_task_performance(self):
        """Test evaluation of task performance."""
        result = self.evaluator.evaluate_task_performance(self.state)
        
        # Verify result structure
        self.assertIn("eval_id", result)
        self.assertIn("timestamp", result)
        self.assertIn("metrics", result)
        self.assertIn("team_metrics", result)
        self.assertIn("target_achievement", result)
        self.assertIn("summary", result)
        
        # Verify metrics
        metrics = result["metrics"]
        self.assertEqual(metrics["total_tasks"], 10)
        self.assertEqual(metrics["success_rate"], 1.0)
        self.assertGreater(metrics["avg_quality"], 0.7)
        self.assertEqual(metrics["deadline_met_rate"], 1.0)
        
        # Verify team metrics
        team_metrics = result["team_metrics"]
        self.assertIn("research", team_metrics)
        self.assertIn("writing", team_metrics)
        self.assertEqual(team_metrics["research"]["task_count"], 5)
        self.assertEqual(team_metrics["writing"]["task_count"], 5)
        
        # Verify target achievement
        target_achievement = result["target_achievement"]
        self.assertIn("success_rate", target_achievement)
        self.assertIn("avg_quality", target_achievement)
        
        # Test with empty state
        empty_state = State()
        result = self.evaluator.evaluate_task_performance(empty_state)
        self.assertIn("Insufficient data", result["summary"])

    def test_evaluate_code_improvements(self):
        """Test evaluation of code improvements."""
        # Mock baseline metrics
        self.evaluator.eval_metrics = {
            "baseline": {
                "timestamp": time.time() - 7200,
                "overall_score": 0.7,
                "success_rate": 0.8,
                "avg_quality": 0.6,
                "deadline_met_rate": 0.7,
                "avg_task_size": 1.0
            }
        }
        
        # Add current metrics
        with patch.object(self.evaluator, "evaluate_task_performance") as mock_eval:
            mock_eval.return_value = {
                "metrics": {
                    "overall_score": 0.85,
                    "success_rate": 1.0,
                    "avg_quality": 0.8,
                    "deadline_met_rate": 0.9,
                    "avg_task_size": 1.2
                }
            }
            
            result = self.evaluator.evaluate_code_improvements(self.state, "baseline")
            
            # Verify result structure
            self.assertIn("eval_id", result)
            self.assertIn("timestamp", result)
            self.assertIn("baseline_eval_id", result)
            self.assertIn("improvement_scores", result)
            self.assertIn("complexity_factor", result)
            self.assertIn("overall_improvement", result)
            self.assertIn("changes", result)
            
            # Verify improvement scores
            improvement_scores = result["improvement_scores"]
            self.assertIn("overall_score", improvement_scores)
            self.assertIn("success_rate", improvement_scores)
            self.assertIn("avg_quality", improvement_scores)
            
            # Verify complexity adjustment
            self.assertGreater(result["complexity_factor"], 1.0)
            
            # Verify changes list
            self.assertEqual(len(result["changes"]), 2)
            
            # Test with no baseline
            result = self.evaluator.evaluate_code_improvements(self.state)
            self.assertIn("baseline_eval_id", result)
            
            # Test with no code changes
            empty_state = State()
            result = self.evaluator.evaluate_code_improvements(empty_state)
            self.assertIn("No code improvements", result["summary"])

    def test_evaluate_resource_scaling(self):
        """Test evaluation of resource scaling."""
        # Set up metrics for before/after scaling
        metrics_before = [
            TaskMetrics(
                task_id=f"before_{i}",
                team_name="research",
                response_quality=0.6,
                success=True,
                start_time=time.time() - 7200,
                end_time=time.time() - 7150,
                deadline_met=False
            ) for i in range(5)
        ]
        
        metrics_after = [
            TaskMetrics(
                task_id=f"after_{i}",
                team_name="research",
                response_quality=0.8,
                success=True,
                start_time=time.time() - 1800,
                end_time=time.time() - 1750,
                deadline_met=True
            ) for i in range(5)
        ]
        
        self.state.metrics = metrics_before + metrics_after
        
        result = self.evaluator.evaluate_resource_scaling(self.state)
        
        # Verify result structure
        self.assertIn("eval_id", result)
        self.assertIn("timestamp", result)
        self.assertIn("team_scaling", result)
        self.assertIn("overall_effectiveness", result)
        self.assertIn("summary", result)
        
        # Verify team scaling metrics
        team_scaling = result.get("team_scaling", {})
        if "research" in team_scaling:  # This might be empty depending on timestamp calculation
            self.assertIn("before_performance", team_scaling["research"])
            self.assertIn("after_performance", team_scaling["research"])
            self.assertIn("efficiency_change", team_scaling["research"])
        
        # Test with no resource requests
        empty_state = State()
        result = self.evaluator.evaluate_resource_scaling(empty_state)
        self.assertIn("No resource scaling", result["summary"])

    @patch("agent.evaluation.ChatPromptTemplate")
    def test_generate_evaluation_report(self, mock_prompt):
        """Test generation of comprehensive evaluation reports."""
        # Mock LLM response
        mock_llm_chain = MagicMock()
        mock_llm_chain.invoke.return_value = json.dumps({
            "overall_assessment": "The system is performing well with improvements.",
            "strengths": ["Good task quality", "Effective resource scaling"],
            "weaknesses": ["Occasional deadline misses"],
            "improvement_recommendations": ["Further optimize research team allocation"]
        })
        
        # Configure mocks
        mock_prompt.from_template.return_value = MagicMock()
        
        # Patch internal evaluation methods
        with patch.object(self.evaluator, "evaluate_task_performance") as mock_task_eval, \
             patch.object(self.evaluator, "evaluate_code_improvements") as mock_code_eval, \
             patch.object(self.evaluator, "evaluate_resource_scaling") as mock_resource_eval, \
             patch("langchain_core.runnables.RunnableLambda", MagicMock()), \
             patch("langchain_core.runnables.RunnablePassthrough", MagicMock()):
            
            # Configure mock returns
            mock_task_eval.return_value = {
                "metrics": {"overall_score": 0.85},
                "summary": "Good performance overall"
            }
            mock_code_eval.return_value = {
                "overall_improvement": 0.2,
                "summary": "Code improvements enhanced performance"
            }
            mock_resource_eval.return_value = {
                "overall_effectiveness": 0.15,
                "summary": "Resource scaling was effective"
            }
            
            # Force LLM chain creation and response
            self.evaluator.llm = MagicMock()
            self.evaluator.llm.pipe.return_value = mock_llm_chain
            
            # Generate report
            result = self.evaluator.generate_evaluation_report(self.state, {})
            
            # Verify report structure
            self.assertIn("report_id", result)
            self.assertIn("timestamp", result)
            self.assertIn("performance", result)
            self.assertIn("code_improvements", result)
            self.assertIn("resource_scaling", result)
            self.assertIn("analysis", result)
            self.assertIn("summary", result)
            
            # Verify analysis content
            analysis = result["analysis"]
            self.assertIn("overall_assessment", analysis)
            self.assertIn("strengths", analysis)
            self.assertIn("weaknesses", analysis)
            self.assertIn("improvement_recommendations", analysis)


if __name__ == "__main__":
    unittest.main()