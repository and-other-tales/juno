# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Hierarchical agent teams implementation using LangGraph.

This module creates a hierarchical multi-agent system with:
1. Top-level supervisor to coordinate between specialized teams
2. Research team (search agent + web scraper agent)
3. Writing team (note taker + document writer)
4. Juno team (evaluator + code agent) for monitoring and improving the system
5. Autonomous task generation and review with supervisor grading
"""

import os
import time
from typing import Dict, Any, Annotated, Union, Literal, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START, END
from langgraph.types import Command, InjectedState

from agent.configuration import Configuration
from agent.state import State, TaskMetrics, AgentPerformance
from agent.teams.research import create_research_team
from agent.teams.writing import create_writing_team
from agent.teams.juno import create_juno_team
from agent.task_generator import update_state_for_new_cycle
from agent.review import update_state_with_review
from agent.supervisor_feedback import process_supervisor_feedback
from agent.utils import make_supervisor_node, process_team_output


def configure_environment(config: Configuration) -> None:
    """Configure environment variables based on the configuration."""
    if config.openai_api_key:
        os.environ["OPENAI_API_KEY"] = config.openai_api_key
    
    if config.tavily_api_key:
        os.environ["TAVILY_API_KEY"] = config.tavily_api_key


def create_performance_metric(
    team_name: str,
    agent_name: str,
    task_description: str,
    start_time: float,
    success: bool = True,
    error_message: Optional[str] = None,
) -> TaskMetrics:
    """Create a performance metric for task execution."""
    metric = TaskMetrics(
        start_time=start_time,
        end_time=time.time(),
        task_id=f"{team_name}-{agent_name}-{time.time()}",
        task_description=task_description,
        agent_name=agent_name,
        team_name=team_name,
        success=success,
        error_message=error_message
    )
    
    return metric


def create_research_team_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
    """Call the research team and process the result."""
    configuration = Configuration.from_runnable_config(config)
    configure_environment(configuration)
    
    start_time = time.time()
    
    try:
        # Create the research team
        research_team = create_research_team(configuration)
        
        # Invoke the research team
        result = research_team.invoke(
            {"messages": [state["messages"][-1]]},
            config={"recursion_limit": configuration.recursion_limit},
        )
        
        # Process the result
        updated_state = process_team_output(state, result)
        updated_state["next"] = "supervisor"
        updated_state["research_result"] = result["messages"][-1].content if result.get("messages") else None
        
        # Add performance metric
        metric = create_performance_metric(
            team_name="research",
            agent_name="team",
            task_description=f"Research for: {state.get('current_task', 'Unknown task')}",
            start_time=start_time,
            success=True
        )
        
        metrics = list(updated_state.get("metrics", []))
        metrics.append(metric)
        updated_state["metrics"] = metrics
        
        # Have the supervisor grade the result
        updated_state = process_supervisor_feedback(updated_state, config)
        
        return updated_state
    
    except Exception as e:
        # Handle errors
        error_message = f"Error in research team: {str(e)}"
        
        # Add error metric
        metric = create_performance_metric(
            team_name="research",
            agent_name="team",
            task_description=f"Research for: {state.get('current_task', 'Unknown task')}",
            start_time=start_time,
            success=False,
            error_message=error_message
        )
        
        metrics = list(state.get("metrics", []))
        metrics.append(metric)
        
        # Update agent performance
        agent_performances = dict(state.get("agent_performances", {}))
        if "research" not in agent_performances:
            agent_performances["research"] = AgentPerformance(
                agent_id="research",
                team_name="research"
            )
        agent_performances["research"].error_count += 1
        
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=error_message)],
            "next": "supervisor",
            "metrics": metrics,
            "agent_performances": agent_performances
        }


def create_writing_team_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
    """Call the writing team and process the result."""
    configuration = Configuration.from_runnable_config(config)
    configure_environment(configuration)
    
    start_time = time.time()
    
    try:
        # Create the writing team
        writing_team = create_writing_team(configuration)
        
        # Build message with research context if available
        messages = list(state["messages"])
        last_message = messages[-1] if messages else None
        
        if last_message and state.get("research_result"):
            research_context = f"{last_message.content}\n\nResearch context: {state['research_result']}"
            input_message = HumanMessage(content=research_context)
        else:
            input_message = last_message
        
        # Invoke the writing team
        result = writing_team.invoke(
            {"messages": [input_message] if input_message else []},
            config={"recursion_limit": configuration.recursion_limit},
        )
        
        # Process the result
        updated_state = process_team_output(state, result)
        updated_state["next"] = "supervisor"
        updated_state["writing_result"] = result["messages"][-1].content if result.get("messages") else None
        
        # Add performance metric
        metric = create_performance_metric(
            team_name="writing",
            agent_name="team",
            task_description=f"Writing for: {state.get('current_task', 'Unknown task')}",
            start_time=start_time,
            success=True
        )
        
        metrics = list(updated_state.get("metrics", []))
        metrics.append(metric)
        updated_state["metrics"] = metrics
        
        # Review the result
        updated_state = update_state_with_review(updated_state, config)
        
        # Have the supervisor grade the result
        updated_state = process_supervisor_feedback(updated_state, config)
        
        return updated_state
    
    except Exception as e:
        # Handle errors
        error_message = f"Error in writing team: {str(e)}"
        
        # Add error metric
        metric = create_performance_metric(
            team_name="writing",
            agent_name="team",
            task_description=f"Writing for: {state.get('current_task', 'Unknown task')}",
            start_time=start_time,
            success=False,
            error_message=error_message
        )
        
        metrics = list(state.get("metrics", []))
        metrics.append(metric)
        
        # Update agent performance
        agent_performances = dict(state.get("agent_performances", {}))
        if "writing" not in agent_performances:
            agent_performances["writing"] = AgentPerformance(
                agent_id="writing",
                team_name="writing"
            )
        agent_performances["writing"].error_count += 1
        
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=error_message)],
            "next": "supervisor",
            "metrics": metrics,
            "agent_performances": agent_performances
        }


def create_juno_team_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
    """Call the Juno team for evaluation and code improvement."""
    configuration = Configuration.from_runnable_config(config)
    configure_environment(configuration)
    
    start_time = time.time()
    
    try:
        # Create the Juno team
        juno_team = create_juno_team(configuration)
        
        # Prepare the input message with performance data
        # In a real implementation, we would format the metrics and targets data
        metrics_summary = f"Total metrics: {len(state.get('metrics', []))}"
        targets_summary = f"Performance targets: {len(state.get('performance_targets', []))}"
        
        # Check for teams that need improvement
        agent_performances = state.get("agent_performances", {})
        teams_needing_improvement = []
        for team_name, performance in agent_performances.items():
            if performance.needs_improvement:
                teams_needing_improvement.append(team_name)
        
        # Check for teams with consistently low quality
        team_low_quality_counts = state.get("team_low_quality_counts", {})
        low_quality_teams = [
            team for team, count in team_low_quality_counts.items() 
            if count >= 3
        ]
        
        input_message = HumanMessage(
            content=f"""
            Evaluate system performance and implement necessary improvements.
            
            {metrics_summary}
            {targets_summary}
            
            Teams needing improvement: {', '.join(teams_needing_improvement) if teams_needing_improvement else 'None'}
            Teams with consistently low quality: {', '.join(low_quality_teams) if low_quality_teams else 'None'}
            
            Current task: {state.get('current_task')}
            Completed tasks: {len(state.get('completed_tasks', []))}
            Cycle count: {state.get('cycle_count', 0)} / {configuration.max_cycles}
            
            Issues identified:
            {chr(10).join([f"- {issue}" for issue in state.get('issues_identified', [])[:5]])}
            
            Recent supervisor feedback:
            {chr(10).join([f"- {feedback}" for team, feedbacks in state.get('supervisor_feedback', {}).items() for feedback in feedbacks[-3:]])}
            """
        )
        
        # Invoke the Juno team
        result = juno_team.invoke(
            {"messages": [input_message]},
            config={"recursion_limit": configuration.recursion_limit},
        )
        
        # Process the result
        updated_state = process_team_output(state, result)
        updated_state["next"] = "task_generator"  # Go to task generator after Juno finishes
        updated_state["juno_result"] = result["messages"][-1].content if result.get("messages") else None
        
        # Merge any additional state updates from Juno
        for key in ["issues_identified", "fixes_implemented", "code_changes"]:
            if key in result:
                updated_state[key] = result[key]
        
        # Add performance metric
        metric = create_performance_metric(
            team_name="juno",
            agent_name="team",
            task_description="System evaluation and improvement",
            start_time=start_time,
            success=True
        )
        
        metrics = list(updated_state.get("metrics", []))
        metrics.append(metric)
        updated_state["metrics"] = metrics
        
        # Reset low quality counters if improvements were made
        if teams_needing_improvement or low_quality_teams:
            team_low_quality_counts = dict(updated_state.get("team_low_quality_counts", {}))
            for team in teams_needing_improvement + low_quality_teams:
                team_low_quality_counts[team] = 0
            updated_state["team_low_quality_counts"] = team_low_quality_counts
        
        return updated_state
    
    except Exception as e:
        # Handle errors
        error_message = f"Error in Juno team: {str(e)}"
        
        # Add error metric
        metric = create_performance_metric(
            team_name="juno",
            agent_name="team",
            task_description="System evaluation and improvement",
            start_time=start_time,
            success=False,
            error_message=error_message
        )
        
        metrics = list(state.get("metrics", []))
        metrics.append(metric)
        
        # Update agent performance
        agent_performances = dict(state.get("agent_performances", {}))
        if "juno" not in agent_performances:
            agent_performances["juno"] = AgentPerformance(
                agent_id="juno",
                team_name="juno"
            )
        agent_performances["juno"].error_count += 1
        
        return {
            **state,
            "messages": state["messages"] + [HumanMessage(content=error_message)],
            "next": "task_generator",  # Still proceed to task generator
            "metrics": metrics,
            "agent_performances": agent_performances
        }


def task_generator_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
    """Generate a new task for the next cycle."""
    configuration = Configuration.from_runnable_config(config)
    
    if not configuration.auto_generate_tasks:
        # If auto-generation is disabled, end the workflow
        return {
            **state,
            "next": "__end__"
        }
    
    # Update the state for a new cycle
    updated_state = update_state_for_new_cycle(state, configuration)
    
    # Return with the updated state
    return updated_state


# Create hierarchical agent system
workflow = StateGraph(State, config_schema=Configuration)

# Initialize the main supervisor
def create_top_supervisor_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
    """Main supervisor node that coordinates between teams."""
    configuration = Configuration.from_runnable_config(config)
    
    # Initialize model
    llm = init_chat_model(
        configuration.model_name,
        model_provider=configuration.model_provider,
    )
    
    # Create supervisor function
    supervisor_node = make_supervisor_node(
        llm,
        members=["research_team", "writing_team", "juno_team", "task_generator"],
        system_prompt=(
            "You are the lead supervisor managing a hierarchical team of AI agents. "
            "You have specialized teams at your disposal:\n"
            "1. RESEARCH_TEAM: For finding and retrieving information from the web\n"
            "2. WRITING_TEAM: For creating outlines and written documents\n"
            "3. JUNO_TEAM: For evaluating performance and implementing improvements\n"
            "4. TASK_GENERATOR: For creating new tasks automatically\n\n"
            "Your workflow for each task should generally follow these steps:\n"
            "1. Send to RESEARCH_TEAM to gather information\n"
            "2. Send to WRITING_TEAM to create documents based on research\n"
            "3. When quality issues occur consistently, send to JUNO_TEAM\n"
            "4. Send to TASK_GENERATOR to start a new task cycle\n\n"
            "Based on the current state, delegate work to the appropriate team. "
            "When you believe the entire process should end, respond with __end__."
        )
    )
    
    # Get the initial state
    if not state["messages"]:
        # If there are no messages yet, generate the first task
        if configuration.auto_generate_tasks:
            return {"next": "task_generator"}
        else:
            # If auto-generation is disabled, wait for the user
            return {"next": None}
    
    # Check if we need to route to Juno based on agent performance
    agent_performances = state.get("agent_performances", {})
    team_low_quality_counts = state.get("team_low_quality_counts", {})
    
    for team_name, performance in agent_performances.items():
        if performance.needs_improvement:
            return {"next": "juno_team"}
    
    for team_name, count in team_low_quality_counts.items():
        if count >= 3:
            return {"next": "juno_team"}
    
    # Execute the supervisor
    result = supervisor_node(state, config)
    return result


# Add nodes to the graph
workflow.add_node("supervisor", create_top_supervisor_node)
workflow.add_node("research_team", create_research_team_node)
workflow.add_node("writing_team", create_writing_team_node)
workflow.add_node("juno_team", create_juno_team_node)
workflow.add_node("task_generator", task_generator_node)

# Add edges
workflow.add_edge(START, "supervisor")
workflow.add_conditional_edges(
    "supervisor",
    lambda state: state.get("next") or "research_team", 
    {
        "research_team": "research_team",
        "writing_team": "writing_team",
        "juno_team": "juno_team",
        "task_generator": "task_generator",
        "__end__": END,
    },
)
workflow.add_edge("research_team", "supervisor")
workflow.add_edge("writing_team", "supervisor")
workflow.add_edge("juno_team", "supervisor")
workflow.add_edge("task_generator", "supervisor")

# Compile the workflow
graph = workflow.compile()
graph.name = "Self-improving Hierarchical Agent Teams"  # This defines the custom name in LangSmith