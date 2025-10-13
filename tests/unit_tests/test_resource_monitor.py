# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Unit tests for the resource monitor."""

import time
import unittest
from unittest.mock import patch, MagicMock

from agent.state import State, TaskMetrics
from agent.resource_monitor import (
    monitor_new_resource,
    calculate_team_performance,
    calculate_efficiency_change,
    create_resource_monitoring_report
)


class TestResourceMonitor(unittest.TestCase):
    """Test cases for the resource monitor functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.state = State()
        
        # Create test metrics
        self.old_metrics = [
            TaskMetrics(
                task_id=f"old_task_{i}",
                team_name="research",
                response_quality=0.6,
                success=True,
                duration=10.0,
                deadline_met=False
            ) for i in range(5)
        ]
        
        self.new_metrics = [
            TaskMetrics(
                task_id=f"new_task_{i}",
                team_name="research",
                response_quality=0.8,
                success=True,
                duration=8.0,
                deadline_met=True
            ) for i in range(5)
        ]
        
        # Set up resource change data
        self.resource_change = {
            "team": "research",
            "current_agents": 1,
            "recommended_agents": 2,
            "timestamp": time.time() - 3600  # 1 hour ago
        }

    def test_calculate_team_performance(self):
        """Test calculation of team performance metrics."""
        # Set state with metrics
        self.state.metrics = self.old_metrics
        self.state.team_resources = {"research": MagicMock(current_agents=1)}
        
        # Calculate performance
        performance = calculate_team_performance(self.state, "research", 1)
        
        # Verify metrics
        self.assertEqual(performance["avg_quality"], 0.6)
        self.assertEqual(performance["success_rate"], 1.0)
        self.assertEqual(performance["deadline_met_rate"], 0.0)
        
        # Test with no metrics
        self.state.metrics = []
        performance = calculate_team_performance(self.state, "research", 1)
        self.assertEqual(performance["avg_quality"], 0.0)
        
        # Test with non-existent team
        self.state.metrics = self.old_metrics
        performance = calculate_team_performance(self.state, "non_existent", 1)
        self.assertEqual(performance["avg_quality"], 0.0)

    def test_calculate_efficiency_change(self):
        """Test calculation of efficiency change after scaling."""
        # Set up performance data
        before = {
            "avg_quality": 0.6,
            "success_rate": 0.8,
            "deadline_met_rate": 0.7
        }
        
        after = {
            "avg_quality": 0.9,
            "success_rate": 1.0,
            "deadline_met_rate": 0.9
        }
        
        # Test with improvement greater than resource increase
        efficiency = calculate_efficiency_change(before, after, 1, 2)
        self.assertGreater(efficiency, 0)
        
        # Test with improvement equal to resource increase
        less_improvement = {
            "avg_quality": 0.7,
            "success_rate": 0.9,
            "deadline_met_rate": 0.8
        }
        efficiency = calculate_efficiency_change(before, less_improvement, 1, 2)
        self.assertLess(efficiency, 0)
        
        # Test with no improvement
        no_improvement = {
            "avg_quality": 0.6,
            "success_rate": 0.8,
            "deadline_met_rate": 0.7
        }
        efficiency = calculate_efficiency_change(before, no_improvement, 1, 2)
        self.assertLess(efficiency, 0)
        
        # Test with zero values
        zero_before = {"avg_quality": 0, "success_rate": 0, "deadline_met_rate": 0}
        efficiency = calculate_efficiency_change(zero_before, after, 1, 2)
        self.assertGreater(efficiency, 0)

    def test_monitor_new_resource(self):
        """Test monitoring of newly added resources."""
        # Setup state with before/after metrics
        self.state.metrics = self.old_metrics + self.new_metrics
        self.state.team_resources = {"research": MagicMock(current_agents=2)}
        
        # Patch calculate_team_performance to return controlled values
        with patch("agent.resource_monitor.calculate_team_performance") as mock_calc:
            mock_calc.side_effect = [
                # First call for old agent count
                {
                    "avg_quality": 0.6,
                    "success_rate": 0.8,
                    "deadline_met_rate": 0.7
                },
                # Second call for new agent count
                {
                    "avg_quality": 0.9,
                    "success_rate": 1.0,
                    "deadline_met_rate": 0.9
                }
            ]
            
            # Monitor resource
            success, comments, efficiency_change = monitor_new_resource(
                self.state, "research", 1, 2
            )
            
            # Verify results
            self.assertTrue(success)
            self.assertGreater(efficiency_change, 0)
            self.assertIn("successful", comments)

    def test_create_resource_monitoring_report(self):
        """Test creation of resource monitoring reports."""
        # Setup state with metrics
        self.state.metrics = self.old_metrics + self.new_metrics
        self.state.team_resources = {"research": MagicMock(current_agents=2)}
        
        # Create report
        with patch("agent.resource_monitor.monitor_new_resource") as mock_monitor:
            mock_monitor.return_value = (True, "Resource scaling was highly successful.", 0.3)
            
            report = create_resource_monitoring_report(
                self.state, "research", self.resource_change
            )
            
            # Verify report content
            self.assertEqual(report.name, "resource_monitor")
            self.assertIn("Resource Change Monitoring Report", report.content)
            self.assertIn("Resource scaling was highly successful", report.content)


if __name__ == "__main__":
    unittest.main()