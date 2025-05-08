"""Resource monitoring and testing for the Juno system."""

import time
from typing import Dict, Any, List, Optional, Tuple

from langchain_core.messages import HumanMessage, AIMessage
from agent.state import State, ResourceConfig


def monitor_new_resource(
    state: State,
    team_name: str,
    old_agent_count: int,
    new_agent_count: int
) -> Tuple[bool, str, float]:
    """Monitor and evaluate the performance of newly added resources.
    
    Args:
        state: The current state
        team_name: The name of the team with new resources
        old_agent_count: Previous number of agents
        new_agent_count: New number of agents
        
    Returns:
        Tuple of (success, comments, efficiency_change)
    """
    # Calculate performance metrics before and after the resource change
    performance_before = calculate_team_performance(state, team_name, old_agent_count)
    performance_after = calculate_team_performance(state, team_name, new_agent_count)
    
    # Calculate the efficiency change
    efficiency_change = calculate_efficiency_change(
        performance_before, 
        performance_after, 
        old_agent_count,
        new_agent_count
    )
    
    # Generate comments based on the efficiency change
    success = efficiency_change > 0
    if efficiency_change > 0.2:
        comments = (
            f"Resource scaling for {team_name} team was highly successful. "
            f"Efficiency improved by {efficiency_change:.1%}. "
            f"The additional resources have significantly improved performance."
        )
    elif efficiency_change > 0:
        comments = (
            f"Resource scaling for {team_name} team was modestly successful. "
            f"Efficiency improved by {efficiency_change:.1%}. "
            f"The additional resources have slightly improved performance."
        )
    elif efficiency_change > -0.1:
        comments = (
            f"Resource scaling for {team_name} team had neutral impact. "
            f"Efficiency changed by {efficiency_change:.1%}. "
            f"The additional resources did not significantly affect performance."
        )
    else:
        comments = (
            f"Resource scaling for {team_name} team was inefficient. "
            f"Efficiency decreased by {abs(efficiency_change):.1%}. "
            f"Consider optimizing or reverting the resource allocation."
        )
    
    return success, comments, efficiency_change


def calculate_team_performance(
    state: State,
    team_name: str,
    agent_count: int
) -> Dict[str, float]:
    """Calculate performance metrics for a team with a specific agent count.
    
    Args:
        state: The current state
        team_name: The name of the team
        agent_count: Number of agents
        
    Returns:
        Dictionary of performance metrics
    """
    # Get all metrics where the team had the specific agent count
    metrics = [
        m for m in state.get("metrics", [])
        if m.team_name == team_name
        and state.get("team_resources", {}).get(team_name, {}).get("current_agents", 1) == agent_count
    ]
    
    # If no metrics found, return default metrics
    if not metrics:
        return {
            "avg_quality": 0.0,
            "success_rate": 0.0,
            "avg_duration": 0.0,
            "deadline_met_rate": 0.0
        }
    
    # Calculate metrics
    avg_quality = sum(m.response_quality for m in metrics) / len(metrics) if metrics else 0.0
    success_count = sum(1 for m in metrics if m.success) if metrics else 0
    success_rate = success_count / len(metrics) if metrics else 0.0
    avg_duration = sum(m.duration for m in metrics) / len(metrics) if metrics else 0.0
    
    # Calculate deadline metrics if available
    deadline_met_count = sum(1 for m in metrics if getattr(m, "deadline_met", True)) if metrics else 0
    deadline_met_rate = deadline_met_count / len(metrics) if metrics else 0.0
    
    return {
        "avg_quality": avg_quality,
        "success_rate": success_rate,
        "avg_duration": avg_duration,
        "deadline_met_rate": deadline_met_rate
    }


def calculate_efficiency_change(
    before: Dict[str, float],
    after: Dict[str, float],
    old_agent_count: int,
    new_agent_count: int
) -> float:
    """Calculate the efficiency change after adding resources.
    
    Args:
        before: Performance metrics before the change
        after: Performance metrics after the change
        old_agent_count: Previous number of agents
        new_agent_count: New number of agents
        
    Returns:
        Efficiency change as a float (positive is better)
    """
    # Avoid division by zero
    if old_agent_count == 0 or new_agent_count == 0:
        return 0.0
    
    # Calculate resource ratio
    resource_ratio = new_agent_count / old_agent_count
    
    # Calculate performance improvement factors
    quality_ratio = after["avg_quality"] / before["avg_quality"] if before["avg_quality"] > 0 else 1.0
    success_ratio = after["success_rate"] / before["success_rate"] if before["success_rate"] > 0 else 1.0
    
    # Speed improvement (lower duration is better)
    speed_ratio = (before["avg_duration"] / after["avg_duration"]) if after["avg_duration"] > 0 else 1.0
    
    # Deadline improvement
    deadline_ratio = after["deadline_met_rate"] / before["deadline_met_rate"] if before["deadline_met_rate"] > 0 else 1.0
    
    # Calculate the weighted overall performance change
    performance_change = (
        quality_ratio * 0.3 +
        success_ratio * 0.2 +
        speed_ratio * 0.3 +
        deadline_ratio * 0.2
    )
    
    # Efficiency change factors in both performance and resource costs
    # Higher is better: performance_change / resource_ratio > 1 means we got more performance than we added in resources
    efficiency_change = (performance_change / resource_ratio) - 1.0
    
    return efficiency_change


def create_resource_monitoring_report(
    state: State,
    team_name: str,
    resource_change: Dict[str, Any]
) -> HumanMessage:
    """Create a report on the effectiveness of a resource change.
    
    Args:
        state: The current state
        team_name: The name of the team with resource changes
        resource_change: Details of the resource change
        
    Returns:
        A message with the monitoring report
    """
    old_agent_count = resource_change.get("current_agents", 1)
    new_agent_count = resource_change.get("recommended_agents", old_agent_count)
    
    # Monitor the resource change
    success, comments, efficiency_change = monitor_new_resource(
        state, team_name, old_agent_count, new_agent_count
    )
    
    # Generate a recommendation
    if efficiency_change > 0.1:
        recommendation = "Keep the new resource allocation."
    elif efficiency_change > -0.1:
        recommendation = "Continue monitoring the resource allocation."
    else:
        recommendation = "Consider reverting to the previous resource allocation."
    
    # Create the report
    return HumanMessage(
        content=f"""
        ## Resource Change Monitoring Report
        
        Team: {team_name}
        Previous agent count: {old_agent_count}
        New agent count: {new_agent_count}
        Change timestamp: {time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(resource_change.get("timestamp", time.time())))}
        
        ### Performance Analysis
        
        Efficiency change: {efficiency_change:.1%}
        Status: {"✅ Success" if success else "❌ Suboptimal"}
        
        ### Comments
        
        {comments}
        
        ### Recommendation
        
        {recommendation}
        """,
        name="resource_monitor"
    )