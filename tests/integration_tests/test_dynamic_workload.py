# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Integration tests for dynamic workload and resource scaling."""

import time
import pytest
from unittest.mock import patch, MagicMock
from langsmith import unit

from agent.configuration import Configuration
from agent.graph import create_graph
from agent.state import State, TaskMetrics


@pytest.mark.asyncio
@unit
async def test_graph_with_dynamic_workload() -> None:
    """Test that the graph works with dynamic workload enabled."""
    config = Configuration()
    config.enable_dynamic_workload = True
    config.random_workload_increase = 0.5
    config.resource_scaling = True
    
    graph = create_graph(config)
    compiled_graph = graph.compile()
    
    # Create state with task
    state = State()
    state.messages = [{"role": "human", "content": "Research quantum computing"}]
    state.current_task = "Research quantum computing"
    state.enable_dynamic_workload = True
    
    # Path workload manager to ensure controlled testing
    with patch("agent.workload_manager.random_workload_increase") as mock_increase, \
         patch("agent.workload_manager.set_task_deadline") as mock_deadline, \
         patch("agent.supervisor_feedback.grade_team_output") as mock_grade:
        
        # Configure mocks
        mock_increase.return_value = (True, 1.5)
        mock_deadline.return_value = time.time() + 600
        mock_grade.return_value = (0.8, "Good output", ["Minor formatting issues"])
        
        # Add a result to be evaluated
        state.research_result = "Quantum computing uses quantum mechanics principles..."
        
        # Invoke the graph
        result = await compiled_graph.ainvoke(state)
        
        # Verify workload adjustment was applied
        assert result.get("current_task_size", 0) == 1.5
        assert result.get("current_task_deadline", 0) > time.time()


@pytest.mark.asyncio
@unit
async def test_graph_with_resource_scaling() -> None:
    """Test that the graph correctly routes to Juno team when resources are needed."""
    config = Configuration()
    config.resource_scaling = True
    
    graph = create_graph(config)
    compiled_graph = graph.compile()
    
    # Create state with resource constraints
    state = State()
    state.messages = [{"role": "human", "content": "Research quantum computing"}]
    state.current_task = "Research quantum computing"
    state.resource_scaling_enabled = True
    
    # Add metrics showing missed deadlines
    metrics = [
        TaskMetrics(
            task_id=f"task_{i}",
            team_name="research",
            response_quality=0.6,
            deadline=time.time() - 100,  # Past deadline
            deadline_met=False
        ) for i in range(5)
    ]
    state.metrics = metrics
    state.missed_deadlines_count = 5
    
    # Add resource config
    state.team_resources = {"research": MagicMock(current_agents=1, max_agents=3)}
    
    # Patch functions to control flow
    with patch("agent.workload_manager.evaluate_resource_needs") as mock_evaluate, \
         patch("agent.supervisor_feedback.grade_team_output") as mock_grade:
        
        # Configure mocks
        mock_evaluate.return_value = {
            "team": "research",
            "current_agents": 1,
            "recommended_agents": 2,
            "reason": "High deadline miss rate"
        }
        mock_grade.return_value = (0.7, "Adequate output", ["Some deadline issues"])
        
        # Add a result to be evaluated
        state.research_result = "Quantum computing uses quantum mechanics principles..."
        
        # Invoke the graph
        result = await compiled_graph.ainvoke(state)
        
        # Verify routing to Juno team
        assert result.get("next") == "juno_team"
        assert len(result.get("resource_change_requests", [])) == 1
        assert result["resource_change_requests"][0]["team"] == "research"


@pytest.mark.asyncio
@unit
async def test_full_cycle_with_evaluation() -> None:
    """Test a complete cycle with evaluation."""
    config = Configuration()
    config.enable_dynamic_workload = True
    config.resource_scaling = True
    
    graph = create_graph(config)
    compiled_graph = graph.compile()
    
    # Create state with completed cycle
    state = State()
    state.messages = [{"role": "human", "content": "Research quantum computing"}]
    state.current_task = "Research quantum computing"
    state.cycle_count = 3
    
    # Add metrics for performance evaluation
    metrics = []
    for i in range(10):
        metric = TaskMetrics(
            task_id=f"task_{i}",
            team_name="research" if i % 2 == 0 else "writing",
            response_quality=0.7 + (i * 0.02),
            success=True,
            start_time=time.time() - 3600,
            end_time=time.time() - 3550,
            duration=50.0,
            deadline=time.time() - 3500,
            deadline_met=True,
            task_size=1.0
        )
        metrics.append(metric)
    
    state.metrics = metrics
    
    # Add code improvements
    state.fixes_implemented = ["Improved task routing", "Fixed deadline calculation"]
    state.code_changes = {
        "change_1": {
            "issues_fixed": ["Routing inefficiency"],
            "implemented_fixes": ["Improved task routing"],
            "timestamp": time.time() - 3600
        }
    }
    
    # Mock functions to isolate testing
    with patch("agent.task_generator.generate_task") as mock_generate, \
         patch("agent.evaluation.JunoEvaluator.generate_evaluation_report") as mock_eval:
        
        # Configure mocks
        mock_generate.return_value = "Analyze quantum computing applications in cryptography"
        mock_eval.return_value = {
            "report_id": "test_report",
            "summary": "System is performing well with effective resource utilization",
            "performance": {"metrics": {"overall_score": 0.85}},
            "analysis": {
                "overall_assessment": "System performs well",
                "improvement_recommendations": ["Further optimize resource allocation"]
            }
        }
        
        # Invoke the graph to complete the cycle
        result = await compiled_graph.ainvoke(state)
        
        # Verify new cycle started with evaluation
        assert result.get("cycle_count", 0) > state.cycle_count
        assert isinstance(result.get("current_task"), str)