"""Unit tests for the workload manager."""

import time
import unittest
from unittest.mock import patch, MagicMock

from agent.configuration import Configuration
from agent.state import State, TaskMetrics
from agent.workload_manager import (
    random_workload_increase,
    set_task_deadline,
    evaluate_resource_needs,
    apply_workload_adjustments
)


class TestWorkloadManager(unittest.TestCase):
    """Test cases for the workload manager functions."""

    def setUp(self):
        """Set up test fixtures."""
        self.config = Configuration()
        self.config.enable_dynamic_workload = True
        self.config.random_workload_increase = 0.5
        self.config.max_task_size_multiplier = 2.0
        self.config.default_deadline_minutes = 10
        
        self.state = State()
        self.state.current_task = "Test task"
        self.state.current_task_size = 1.0
        self.state.enable_dynamic_workload = True
        self.state.resource_scaling_enabled = True

    @patch("random.random")
    def test_random_workload_increase(self, mock_random):
        """Test the random workload increase function."""
        # Test when random value is below threshold (should increase)
        mock_random.return_value = 0.4
        result, new_size = random_workload_increase(self.state, self.config)
        self.assertTrue(result)
        self.assertGreater(new_size, 1.0)
        
        # Test when random value is above threshold (should not increase)
        mock_random.return_value = 0.6
        result, new_size = random_workload_increase(self.state, self.config)
        self.assertFalse(result)
        self.assertEqual(new_size, 1.0)
        
        # Test when dynamic workload is disabled
        self.config.enable_dynamic_workload = False
        result, new_size = random_workload_increase(self.state, self.config)
        self.assertFalse(result)
        self.assertEqual(new_size, 1.0)
        
        # Test when current size is already at max
        self.config.enable_dynamic_workload = True
        self.state.current_task_size = 2.0
        result, new_size = random_workload_increase(self.state, self.config)
        self.assertFalse(result)
        self.assertEqual(new_size, 2.0)

    def test_set_task_deadline(self):
        """Test setting task deadlines."""
        # Test with default task size
        current_time = time.time()
        deadline = set_task_deadline(self.state, self.config)
        
        # Expected deadline should be approximately current time + default minutes
        expected_min_time = current_time + (self.config.default_deadline_minutes * 60 * 0.9)  # With 10% variation
        expected_max_time = current_time + (self.config.default_deadline_minutes * 60 * 1.1)
        
        self.assertGreaterEqual(deadline, expected_min_time)
        self.assertLessEqual(deadline, expected_max_time)
        
        # Test with larger task size
        deadline = set_task_deadline(self.state, self.config, task_size=2.0)
        
        # Expected deadline should be approximately current time + (default minutes * 2)
        expected_min_time = current_time + (self.config.default_deadline_minutes * 60 * 2 * 0.9)
        expected_max_time = current_time + (self.config.default_deadline_minutes * 60 * 2 * 1.1)
        
        self.assertGreaterEqual(deadline, expected_min_time)
        self.assertLessEqual(deadline, expected_max_time)

    def test_evaluate_resource_needs(self):
        """Test evaluation of resource needs."""
        # Test with no metrics (should return None)
        result = evaluate_resource_needs(self.state, self.config)
        self.assertIsNone(result)
        
        # Add metrics with missed deadlines
        metrics = [
            TaskMetrics(
                task_id=f"task_{i}",
                team_name="research",
                response_quality=0.6,
                deadline=time.time() - 100  # Past deadline
            ) for i in range(5)
        ]
        
        self.state.metrics = metrics
        self.state.team_resources = {"research": MagicMock(current_agents=1, max_agents=3)}
        
        # Test with metrics indicating resource needs
        result = evaluate_resource_needs(self.state, self.config)
        
        self.assertIsNotNone(result)
        self.assertEqual(result["team"], "research")
        self.assertEqual(result["current_agents"], 1)
        self.assertEqual(result["recommended_agents"], 2)
        
        # Test when scaling is disabled
        self.config.resource_scaling = False
        result = evaluate_resource_needs(self.state, self.config)
        self.assertIsNone(result)
        
        # Test when at max agents
        self.config.resource_scaling = True
        self.state.team_resources = {"research": MagicMock(current_agents=3, max_agents=3)}
        result = evaluate_resource_needs(self.state, self.config)
        self.assertIsNone(result)

    @patch("agent.workload_manager.random_workload_increase")
    @patch("agent.workload_manager.set_task_deadline")
    @patch("agent.workload_manager.evaluate_resource_needs")
    def test_apply_workload_adjustments(self, mock_evaluate, mock_set_deadline, mock_increase):
        """Test applying workload adjustments."""
        # Set up mocks
        mock_increase.return_value = (True, 1.5)
        mock_set_deadline.return_value = time.time() + 600  # 10 minutes from now
        mock_evaluate.return_value = {
            "team": "research",
            "current_agents": 1,
            "recommended_agents": 2,
            "reason": "High deadline miss rate"
        }
        
        # Test with all conditions triggering adjustments
        result = apply_workload_adjustments(self.state, self.config)
        
        # Verify task size was updated
        self.assertEqual(result["current_task_size"], 1.5)
        
        # Verify deadline was set
        self.assertGreater(result["current_task_deadline"], time.time())
        
        # Verify resource request was added
        self.assertEqual(len(result["resource_change_requests"]), 1)
        self.assertEqual(result["resource_change_requests"][0]["team"], "research")
        
        # Verify next routing was set to juno_team
        self.assertEqual(result["next"], "juno_team")
        
        # Test with no current task
        self.state.current_task = None
        result = apply_workload_adjustments(self.state, self.config)
        self.assertEqual(result, self.state)


if __name__ == "__main__":
    unittest.main()