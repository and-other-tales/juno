"""Workload management tools for dynamic scaling and deadline handling."""

import time
import random
import math
from typing import Dict, Any, List, Optional, Tuple

from agent.state import State, TaskMetrics, ResourceConfig
from agent.configuration import Configuration


def random_workload_increase(
    state: State,
    config: Configuration
) -> Tuple[bool, float]:
    """Determine if workload should be randomly increased.
    
    Args:
        state: The current state
        config: Configuration parameters
        
    Returns:
        Tuple of (should_increase, new_size_multiplier)
    """
    # Skip if dynamic workload is disabled
    if not config.enable_dynamic_workload:
        return False, 1.0
    
    # Check random probability
    if random.random() > config.random_workload_increase:
        return False, 1.0
    
    # Calculate a random size increase between 1.0 and max_task_size_multiplier
    current_multiplier = state.get("current_task_size", 1.0)
    max_multiplier = config.max_task_size_multiplier
    
    # Ensure we don't exceed the maximum multiplier
    if current_multiplier >= max_multiplier:
        return False, current_multiplier
    
    # Calculate a new size between current and max
    new_multiplier = min(
        max_multiplier,
        current_multiplier + random.uniform(0.2, 0.5)
    )
    
    return True, round(new_multiplier, 1)


def set_task_deadline(
    state: State,
    config: Configuration,
    task_size: float = 1.0
) -> float:
    """Set a deadline for the current task based on its size and complexity.
    
    Args:
        state: The current state
        config: Configuration parameters
        task_size: Size multiplier for the task (1.0 is standard)
        
    Returns:
        Deadline timestamp
    """
    # Base deadline in seconds (convert from minutes)
    base_deadline_seconds = config.default_deadline_minutes * 60
    
    # Adjust deadline based on task size
    adjusted_deadline_seconds = base_deadline_seconds * task_size
    
    # Add some random variation (±10%)
    variation_factor = random.uniform(0.9, 1.1)
    final_deadline_seconds = adjusted_deadline_seconds * variation_factor
    
    # Calculate the deadline timestamp
    deadline = time.time() + final_deadline_seconds
    
    return deadline


def evaluate_resource_needs(
    state: State,
    config: Configuration
) -> Optional[Dict[str, Any]]:
    """Evaluate if the system needs more resources based on performance and deadlines.
    
    Args:
        state: The current state
        config: Configuration parameters
        
    Returns:
        Resource change request or None if no changes needed
    """
    # Skip if resource scaling is disabled
    if not config.resource_scaling:
        return None
    
    # Get all task metrics from the last few cycles
    recent_metrics = state.get("metrics", [])[-10:]
    if not recent_metrics:
        return None
    
    # Calculate how many deadlines were missed
    missed_deadlines = sum(1 for m in recent_metrics if not getattr(m, "deadline_met", True))
    deadline_miss_rate = missed_deadlines / len(recent_metrics) if recent_metrics else 0
    
    # Check performance metrics
    avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
    avg_quality = sum(m.response_quality for m in recent_metrics) / len(recent_metrics) if recent_metrics else 0
    
    # Determine the team with the most missed deadlines
    team_missed = {}
    for m in recent_metrics:
        if not getattr(m, "deadline_met", True):
            team = m.team_name
            team_missed[team] = team_missed.get(team, 0) + 1
    
    # Only proceed if we have a significant deadline miss rate or quality issues
    if deadline_miss_rate < 0.2 and avg_quality > 0.7:
        return None
    
    # Find the team with the most issues
    problem_team = max(team_missed.items(), key=lambda x: x[1])[0] if team_missed else None
    if not problem_team:
        return None
    
    # Check if we can add more resources to this team
    team_resources = state.get("team_resources", {}).get(problem_team)
    if not team_resources:
        return None
    
    if team_resources.current_agents >= team_resources.max_agents:
        return None
    
    # Create a resource change request
    return {
        "team": problem_team,
        "current_agents": team_resources.current_agents,
        "recommended_agents": team_resources.current_agents + 1,
        "reason": f"High deadline miss rate ({deadline_miss_rate:.1%}) for team {problem_team}",
        "timestamp": time.time(),
    }


def apply_workload_adjustments(state: State, config: Configuration) -> State:
    """Apply workload adjustments including random increases and deadline setting.
    
    Args:
        state: The current state
        config: Configuration parameters
        
    Returns:
        Updated state with workload adjustments
    """
    updated_state = dict(state)
    
    # Only apply adjustments when we have a current task
    if not updated_state.get("current_task"):
        return updated_state
    
    # Check if we should randomly increase workload
    should_increase, new_size = random_workload_increase(updated_state, config)
    if should_increase:
        updated_state["current_task_size"] = new_size
        
        # Add a message about increased workload
        messages = list(updated_state.get("messages", []))
        messages.append({
            "content": f"**NOTICE**: Supervisor has increased the workload. Task size is now {new_size}x standard.",
            "role": "system",
        })
        updated_state["messages"] = messages
    
    # Set a deadline if one is not already set
    if not updated_state.get("current_task_deadline"):
        deadline = set_task_deadline(
            updated_state, 
            config, 
            updated_state.get("current_task_size", 1.0)
        )
        updated_state["current_task_deadline"] = deadline
        
        # Format deadline for display
        deadline_str = time.strftime("%H:%M:%S", time.localtime(deadline))
        
        # Add a message about the deadline
        messages = list(updated_state.get("messages", []))
        messages.append({
            "content": f"**DEADLINE**: This task must be completed by {deadline_str}.",
            "role": "system",
        })
        updated_state["messages"] = messages
    
    # Check if we need more resources
    resource_request = evaluate_resource_needs(updated_state, config)
    if resource_request:
        # Add the resource request
        requests = list(updated_state.get("resource_change_requests", []))
        requests.append(resource_request)
        updated_state["resource_change_requests"] = requests
        
        # Set routing to the Juno team if not already set
        if updated_state.get("next") != "juno_team":
            updated_state["next"] = "juno_team"
            
            # Add a message about the resource request
            messages = list(updated_state.get("messages", []))
            messages.append({
                "content": (
                    f"**RESOURCE REQUEST**: Team {resource_request['team']} requires "
                    f"additional resources. Recommendation: Increase from "
                    f"{resource_request['current_agents']} to {resource_request['recommended_agents']} agents. "
                    f"Reason: {resource_request['reason']}"
                ),
                "role": "system",
            })
            updated_state["messages"] = messages
    
    return updated_state