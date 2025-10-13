# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Define the state structures for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Set, Counter

from langchain_core.messages import BaseMessage, HumanMessage, AIMessage


@dataclass
class TaskMetrics:
    """Metrics for task execution."""
    
    start_time: float = 0.0
    end_time: float = 0.0
    deadline: float = 0.0  # Deadline timestamp
    task_id: str = ""
    task_description: str = ""
    agent_name: str = ""
    team_name: str = ""
    success: bool = True
    error_message: Optional[str] = None
    tokens_used: int = 0
    response_quality: float = 0.0  # 0.0 to 1.0
    task_size: float = 1.0  # Relative size/complexity of task (1.0 is standard)
    
    @property
    def duration(self) -> float:
        """Calculate task duration in seconds."""
        if self.end_time and self.start_time:
            return self.end_time - self.start_time
        return 0.0
    
    @property
    def deadline_met(self) -> bool:
        """Check if the task met its deadline."""
        if not self.deadline:
            return True  # No deadline set
        return self.end_time <= self.deadline
    
    @property
    def deadline_buffer(self) -> float:
        """Calculate time buffer (positive) or overrun (negative) in seconds."""
        if not self.deadline:
            return 0.0  # No deadline set
        return self.deadline - self.end_time


@dataclass
class PerformanceTarget:
    """Performance target for the system."""
    
    metric_name: str
    target_value: float
    current_value: float = 0.0
    description: str = ""
    
    @property
    def is_met(self) -> bool:
        """Check if the target is met."""
        return self.current_value >= self.target_value


@dataclass
class AgentPerformance:
    """Agent performance tracker."""
    
    agent_id: str
    team_name: str
    quality_scores: List[float] = field(default_factory=list)
    error_count: int = 0
    success_count: int = 0
    total_time: float = 0.0
    
    @property
    def avg_quality(self) -> float:
        """Calculate average quality score."""
        if not self.quality_scores:
            return 0.0
        return sum(self.quality_scores) / len(self.quality_scores)
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        total = self.success_count + self.error_count
        if total == 0:
            return 1.0
        return self.success_count / total
    
    @property
    def needs_improvement(self) -> bool:
        """Determine if agent needs improvement."""
        # Needs improvement if:
        # 1. 3 or more consecutive failures, OR
        # 2. Average quality below 0.5, OR
        # 3. Success rate below 0.7
        return (
            self.error_count >= 3 or 
            (len(self.quality_scores) >= 3 and self.avg_quality < 0.5) or
            (total := (self.success_count + self.error_count)) >= 5 and (self.success_count / total) < 0.7
        )


@dataclass
class ResourceConfig:
    """Configuration for team resources."""
    
    team_name: str
    current_agents: int = 1
    min_agents: int = 1
    max_agents: int = 3
    agent_types: List[str] = field(default_factory=list)
    scaling_factor: float = 1.0  # How much each agent improves throughput


@dataclass
class State:
    """Defines the state for the hierarchical agent teams system.
    
    This state structure maintains the message history at each level of the hierarchy,
    as well as metadata for routing between teams and agents.
    """
    
    # Main state elements
    messages: List[BaseMessage] = field(default_factory=list)
    next: Optional[str] = None
    
    # Team outputs and context
    research_result: Optional[str] = None
    writing_result: Optional[str] = None
    juno_result: Optional[str] = None
    
    # Task generation and tracking
    current_task: Optional[str] = None
    current_task_deadline: float = 0.0  # Deadline timestamp for current task
    current_task_size: float = 1.0  # Size/complexity multiplier for current task
    completed_tasks: List[str] = field(default_factory=list)
    task_generation_count: int = 0
    
    # Workload management
    enable_dynamic_workload: bool = False  # Whether to enable dynamic workload changes
    random_workload_increase_probability: float = 0.3  # Probability of increasing workload
    max_task_size_multiplier: float = 2.0  # Maximum task size multiplier
    default_deadline_minutes: int = 30  # Default deadline in minutes
    missed_deadlines_count: int = 0  # Count of missed deadlines
    
    # Resource management
    team_resources: Dict[str, ResourceConfig] = field(default_factory=lambda: {
        "research": ResourceConfig(team_name="research"),
        "writing": ResourceConfig(team_name="writing"),
        "juno": ResourceConfig(team_name="juno")
    })
    resource_scaling_enabled: bool = False  # Whether to enable resource scaling
    resource_change_requests: List[Dict[str, Any]] = field(default_factory=list)
    
    # Workspace for storing document references
    workspace: Dict[str, Any] = field(default_factory=dict)
    
    # Performance monitoring
    metrics: List[TaskMetrics] = field(default_factory=list)
    performance_targets: List[PerformanceTarget] = field(default_factory=list)
    agent_performances: Dict[str, AgentPerformance] = field(default_factory=dict)
    
    # Work quality tracking
    quality_threshold: float = 0.7  # Minimum acceptable quality score
    team_low_quality_counts: Dict[str, int] = field(default_factory=lambda: {"research": 0, "writing": 0})
    
    # Supervisor feedback
    supervisor_feedback: Dict[str, List[str]] = field(default_factory=lambda: {"research": [], "writing": [], "juno": []})
    
    # Code improvement tracking
    issues_identified: List[str] = field(default_factory=list)
    fixes_implemented: List[str] = field(default_factory=list)
    code_changes: Dict[str, Any] = field(default_factory=dict)
    
    # Task review results
    review_scores: Dict[str, float] = field(default_factory=dict)
    review_comments: Dict[str, str] = field(default_factory=dict)
    
    # Cycle tracking
    cycle_count: int = 0
    max_cycles: int = 10