"""Task generator for autonomous workflow."""

import random
import time
import uuid
from typing import List, Dict, Any, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate

from agent.configuration import Configuration
from agent.state import State, PerformanceTarget


def generate_random_task(config: Configuration) -> str:
    """Generate a random task based on the configured categories."""
    categories = config.task_categories
    selected_category = random.choice(categories)
    
    # Create a prompt for generating a specific task in the selected category
    task_generation_prompt = ChatPromptTemplate.from_messages([
        ("system", f"You are a task generator for AI agents. Create a specific, detailed task in the '{selected_category}' category. The task should be challenging but achievable."),
        ("human", "Generate a detailed task. Respond only with the task description, no preamble or additional text.")
    ])
    
    # Initialize the model
    llm = init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
    )
    
    # Generate the task
    task_chain = task_generation_prompt | llm
    response = task_chain.invoke({})
    
    return response.content


def initialize_performance_targets(config: Configuration) -> List[PerformanceTarget]:
    """Initialize performance targets based on configuration."""
    targets = []
    
    for metric_name, target_value in config.performance_targets.items():
        target = PerformanceTarget(
            metric_name=metric_name,
            target_value=target_value,
            current_value=0.0,
            description=get_metric_description(metric_name)
        )
        targets.append(target)
    
    return targets


def get_metric_description(metric_name: str) -> str:
    """Get a description for a performance metric."""
    descriptions = {
        "avg_response_time": "Average time in seconds to complete a task",
        "success_rate": "Percentage of tasks completed successfully",
        "response_quality": "Average quality score of task outputs (0-1)",
        "task_completion_rate": "Percentage of assigned tasks that get completed"
    }
    
    return descriptions.get(metric_name, f"Performance metric: {metric_name}")


def update_state_for_new_cycle(state: State, config: Configuration) -> State:
    """Update the state for a new autonomous cycle."""
    # Generate a new task
    new_task = generate_random_task(config)
    
    # Increment the cycle counter
    cycle_count = state.get("cycle_count", 0) + 1
    
    # Check if we've reached the max cycles
    if cycle_count >= config.max_cycles:
        return {
            **state,
            "messages": state["messages"] + [
                HumanMessage(content=f"Maximum cycle count ({config.max_cycles}) reached. Stopping autonomous execution.")
            ],
            "next": "__end__"
        }
    
    # If the previous task exists, add it to completed tasks
    current_task = state.get("current_task")
    if current_task:
        completed_tasks = list(state.get("completed_tasks", []))
        completed_tasks.append(current_task)
    else:
        completed_tasks = state.get("completed_tasks", [])
    
    # Initialize performance targets if not already set
    if not state.get("performance_targets"):
        performance_targets = initialize_performance_targets(config)
    else:
        performance_targets = state.get("performance_targets")
    
    # Create a task message
    task_message = HumanMessage(content=f"Task #{cycle_count}: {new_task}")
    
    # Update the state
    return {
        **state,
        "messages": state["messages"] + [task_message],
        "current_task": new_task,
        "completed_tasks": completed_tasks,
        "cycle_count": cycle_count,
        "task_generation_count": state.get("task_generation_count", 0) + 1,
        "performance_targets": performance_targets,
        "next": "research_team"  # Default to research team for new tasks
    }