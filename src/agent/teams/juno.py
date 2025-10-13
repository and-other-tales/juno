# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Juno team implementation for monitoring and improving the system."""

import time
import json
import os
import inspect
import uuid
import random
from typing import Dict, Any, List, Optional, Annotated, Callable, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, InjectedState
from langgraph_codeact import create_codeact, EvalCoroutine
from langchain_sandbox import PyodideSandbox

from agent.configuration import Configuration
from agent.state import State, TaskMetrics, PerformanceTarget, ResourceConfig
from agent.utils import make_supervisor_node
from agent.tools import list_documents
from agent.resource_monitor import create_resource_monitoring_report


def create_analytics_tools():
    """Create tools for analyzing system performance."""
    
    @create_react_agent.tool
    def calculate_metrics(metrics: List[TaskMetrics]) -> Dict[str, float]:
        """Calculate aggregate metrics from a list of task metrics."""
        if not metrics:
            return {"error": "No metrics provided"}
        
        total_duration = sum(metric.duration for metric in metrics)
        avg_duration = total_duration / len(metrics) if metrics else 0
        success_rate = sum(1 for m in metrics if m.success) / len(metrics) if metrics else 0
        avg_quality = sum(m.response_quality for m in metrics) / len(metrics) if metrics else 0
        
        # Calculate deadline metrics if available
        deadline_met_count = sum(1 for m in metrics if getattr(m, "deadline_met", True))
        deadline_met_rate = deadline_met_count / len(metrics) if metrics else 1.0
        
        # Calculate average task size if available
        avg_task_size = sum(getattr(m, "task_size", 1.0) for m in metrics) / len(metrics) if metrics else 1.0
        
        return {
            "total_tasks": len(metrics),
            "total_duration": total_duration,
            "avg_duration": avg_duration,
            "success_rate": success_rate,
            "avg_quality": avg_quality,
            "deadline_met_rate": deadline_met_rate,
            "avg_task_size": avg_task_size,
            "tasks_by_team": {team: sum(1 for m in metrics if m.team_name == team) 
                             for team in set(m.team_name for m in metrics if m.team_name)}
        }
    
    @create_react_agent.tool
    def check_performance_targets(
        current_metrics: Dict[str, float], 
        targets: List[PerformanceTarget]
    ) -> Dict[str, Any]:
        """Compare current metrics against performance targets."""
        results = {}
        for target in targets:
            metric_name = target.metric_name
            current_value = current_metrics.get(
                metric_name, 
                current_metrics.get(metric_name.replace("_", ""), 0)
            )
            is_met = current_value >= target.target_value
            results[metric_name] = {
                "target": target.target_value,
                "current": current_value,
                "is_met": is_met,
                "gap": target.target_value - current_value if not is_met else 0
            }
        
        overall_success = all(result["is_met"] for result in results.values())
        priority_issues = [
            name for name, result in results.items() 
            if not result["is_met"] and result["gap"] > 0.1
        ]
        
        return {
            "overall_success": overall_success,
            "metrics": results,
            "priority_issues": priority_issues
        }
    
    @create_react_agent.tool
    def analyze_resource_allocation(
        state: State,
        team_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Analyze resource allocation and recommend changes if needed."""
        # Get team resources
        team_resources = state.get("team_resources", {})
        
        # If a specific team is provided, just analyze that team
        if team_name and team_name in team_resources:
            teams_to_analyze = {team_name: team_resources[team_name]}
        else:
            teams_to_analyze = team_resources
        
        results = {}
        for team, resources in teams_to_analyze.items():
            # Get team metrics
            team_metrics = [m for m in state.get("metrics", []) if m.team_name == team]
            recent_metrics = team_metrics[-10:] if len(team_metrics) > 10 else team_metrics
            
            # Calculate performance metrics
            if recent_metrics:
                avg_duration = sum(m.duration for m in recent_metrics) / len(recent_metrics)
                success_rate = sum(1 for m in recent_metrics if m.success) / len(recent_metrics)
                avg_quality = sum(m.response_quality for m in recent_metrics) / len(recent_metrics)
                deadline_met_rate = sum(1 for m in recent_metrics if getattr(m, "deadline_met", True)) / len(recent_metrics)
            else:
                avg_duration = 0
                success_rate = 0
                avg_quality = 0
                deadline_met_rate = 1.0
            
            # Determine if resources are sufficient
            current_agents = resources.current_agents
            max_agents = resources.max_agents
            
            is_resource_constrained = (
                (deadline_met_rate < 0.8 or avg_quality < 0.7)
                and current_agents < max_agents
            )
            
            # Create recommendation
            results[team] = {
                "current_agents": current_agents,
                "max_agents": max_agents,
                "avg_duration": avg_duration,
                "success_rate": success_rate,
                "avg_quality": avg_quality,
                "deadline_met_rate": deadline_met_rate,
                "is_resource_constrained": is_resource_constrained,
                "recommendation": (
                    f"Increase agents from {current_agents} to {min(current_agents + 1, max_agents)}"
                    if is_resource_constrained else "Maintain current resource allocation"
                )
            }
        
        return {
            "analysis": results,
            "overall_recommendation": any(r["is_resource_constrained"] for r in results.values())
        }
    
    return [calculate_metrics, check_performance_targets, analyze_resource_allocation]


def create_pyodide_eval_fn(sandbox_dir: str = "./sessions") -> EvalCoroutine:
    """Create an eval function that uses PyodideSandbox."""
    sandbox = PyodideSandbox(sandbox_dir, allow_net=True)
    
    async def async_eval_fn(code: str, _locals: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
        """Evaluate code in a sandbox environment."""
        # Create a wrapper function that will execute the code and return locals
        wrapper_code = f"""
def execute():
    try:
        # Execute the provided code
{chr(10).join("        " + line for line in code.strip().split(chr(10)))}
        return locals()
    except Exception as e:
        return {{"error": str(e)}}

execute()
"""
        # Convert functions in _locals to their string representation
        context_setup = ""
        for key, value in _locals.items():
            if callable(value):
                # Get the function's source code
                try:
                    src = inspect.getsource(value)
                    context_setup += f"\n{src}"
                except (TypeError, OSError):
                    # Some built-in functions can't be inspected
                    context_setup += f"\n{key} = None  # Could not get source for {key}"
            else:
                context_setup += f"\n{key} = {repr(value)}"
        
        try:
            # Execute the code and get the result
            response = await sandbox.execute(
                code=context_setup + "\n\n" + wrapper_code,
                session_id=str(uuid.uuid4()),
            )
            
            # Check if execution was successful
            if response.stderr:
                return f"Error during execution: {response.stderr}", {}
            
            # Get the output from stdout
            output = (
                response.stdout
                if response.stdout
                else "<Code ran, no output printed to stdout>"
            )
            result = response.result
            
            # If there was an error in the result, return it
            if isinstance(result, dict) and "error" in result:
                return f"Error during execution: {result['error']}", {}
            
            # Get the new variables by comparing with original locals
            new_vars = {
                k: v
                for k, v in result.items()
                if k not in _locals and not k.startswith("_")
            }
            return output, new_vars
            
        except Exception as e:
            return f"Error during PyodideSandbox execution: {repr(e)}", {}
    
    return async_eval_fn


def create_juno_team(config: Configuration) -> StateGraph:
    """Create the Juno team with evaluator and code improvement agents."""
    
    # Initialize the language model
    llm = init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
    )
    
    # Create the evaluator agent
    evaluator_tools = create_analytics_tools() + [list_documents]
    evaluator_agent = create_react_agent(
        llm, 
        tools=evaluator_tools,
        name="evaluator",
        prompt=(
            "You are an evaluator agent that monitors the system's performance. "
            "Your job is to track metrics, identify performance issues, and determine "
            "when code improvements are needed. You should be objective, data-driven, "
            "and focused on continuous improvement."
        ),
    )
    
    # Create the code agent using CodeAct
    eval_fn = create_pyodide_eval_fn(config.sandbox_directory)
    
    # Define tools for code improvements
    code_tools = [
        # Placeholder for actual tools the code agent would use
        # In a real implementation, these would include file operations, code analysis, etc.
        lambda source_path: f"Source code path: {source_path}",
        lambda metrics_data: f"Metrics data: {metrics_data}"
    ]
    
    code_agent = create_codeact(
        llm,
        tools=code_tools, 
        eval_fn=eval_fn,
        prompt=(
            "You are a code improvement agent that identifies and fixes performance issues. "
            "You can analyze metrics, identify bottlenecks, and implement code changes to "
            "improve the system. Your changes should be well-tested, safe, and lead to "
            "measurable improvements in system performance."
        )
    )
    
    # Create evaluator node
    def evaluator_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
        """Execute the evaluator agent to analyze system performance."""
        start_time = time.time()
        
        # Prepare input with metrics data
        metrics_data = json.dumps([{
            "task_id": m.task_id,
            "duration": m.duration,
            "agent": m.agent_name,
            "team": m.team_name,
            "success": m.success,
            "quality": m.response_quality
        } for m in state.get("metrics", [])])
        
        targets_data = json.dumps([{
            "metric": t.metric_name,
            "target": t.target_value,
            "current": t.current_value,
            "description": t.description
        } for t in state.get("performance_targets", [])])
        
        task_info = f"""
        Current task: {state.get('current_task')}
        Completed tasks: {len(state.get('completed_tasks', []))}
        Cycle count: {state.get('cycle_count', 0)}
        
        Metrics data: {metrics_data}
        
        Performance targets: {targets_data}
        
        Issues identified so far: {state.get('issues_identified', [])}
        Fixes implemented: {state.get('fixes_implemented', [])}
        """
        
        evaluation_request = HumanMessage(
            content=f"Please evaluate the current system performance and identify any issues that need to be fixed:\n\n{task_info}"
        )
        
        result = evaluator_agent.invoke({"messages": [evaluation_request]})
        
        # Create a task metric for this evaluation
        metric = TaskMetrics(
            start_time=start_time,
            end_time=time.time(),
            task_id=f"evaluation-{time.time()}",
            task_description="System performance evaluation",
            agent_name="evaluator",
            team_name="juno"
        )
        
        # Update metrics
        metrics = list(state.get("metrics", []))
        metrics.append(metric)
        
        # Extract issues from the evaluation
        response_content = result["messages"][-1].content
        issues = []
        if "issues" in response_content.lower() or "problems" in response_content.lower():
            for line in response_content.split("\n"):
                if line.strip().startswith("-") or line.strip().startswith("*"):
                    issues.append(line.strip())
        
        # Update the issues identified list
        identified_issues = list(state.get("issues_identified", []))
        identified_issues.extend(issues)
        
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="evaluator")
            ],
            "next": "supervisor",
            "metrics": metrics,
            "issues_identified": identified_issues
        }
    
    # Create code agent node
    def code_agent_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
        """Execute the code agent to implement improvements."""
        start_time = time.time()
        configuration = Configuration.from_runnable_config(config)
        
        # Check if this is a resource scaling request
        resource_request = None
        if state.get("resource_change_requests"):
            resource_request = state["resource_change_requests"][-1]
        
        # Get the latest issues
        issues = state.get("issues_identified", [])
        if not issues and not resource_request:
            return {
                "messages": [
                    HumanMessage(content="No issues to fix.", name="code_agent")
                ],
                "next": "supervisor"
            }
        
        # Handle resource scaling request if present
        if resource_request:
            team_name = resource_request.get("team", "")
            current_agents = resource_request.get("current_agents", 1)
            recommended_agents = resource_request.get("recommended_agents", current_agents + 1)
            reason = resource_request.get("reason", "Performance issues detected")
            
            # Prepare the resource scaling implementation
            scaling_request = HumanMessage(
                content=f"""
                RESOURCE SCALING REQUEST
                
                Please implement the following resource scaling change:
                
                Team: {team_name}
                Current agents: {current_agents}
                Recommended agents: {recommended_agents}
                Reason: {reason}
                
                Implementation steps:
                1. Update the team's resource configuration
                2. Allocate additional agent instances
                3. Configure the new agents with appropriate tools and knowledge
                4. Test the new resource allocation
                5. Monitor performance changes
                """
            )
            
            # In a real implementation, this would execute the CodeAct agent
            # and actually deploy new agent resources
            
            # Simulate implementation of resource scaling
            # In a real system, this would actually allocate new agents
            team_resources = dict(state.get("team_resources", {}))
            if team_name in team_resources:
                resource_config = team_resources[team_name]
                resource_config.current_agents = min(recommended_agents, resource_config.max_agents)
                team_resources[team_name] = resource_config
            
            # Create a monitoring report
            monitoring_report = create_resource_monitoring_report(state, team_name, resource_request)
            
            # Add the report to messages
            messages = [
                HumanMessage(
                    content=f"""
                    RESOURCE SCALING IMPLEMENTED
                    
                    Team: {team_name}
                    Previous agents: {current_agents}
                    New agents: {recommended_agents}
                    
                    Implementation details:
                    - Added {recommended_agents - current_agents} new agent(s) to the {team_name} team
                    - Configured agents with appropriate tools and knowledge
                    - Enabled performance monitoring for the new resources
                    
                    The system will continue to monitor the performance impact of this change.
                    """,
                    name="code_agent"
                ),
                monitoring_report
            ]
            
            # Create a task metric for this resource scaling
            metric = TaskMetrics(
                start_time=start_time,
                end_time=time.time(),
                task_id=f"resource-scaling-{time.time()}",
                task_description=f"Resource scaling for {team_name} team",
                agent_name="code_agent",
                team_name="juno"
            )
            
            # Update metrics
            metrics = list(state.get("metrics", []))
            metrics.append(metric)
            
            # Return the updated state
            return {
                "messages": messages,
                "next": "supervisor",
                "metrics": metrics,
                "team_resources": team_resources
            }
        
        # Handle regular code improvements
        # Prepare input with issues and past fixes
        issues_str = "\n".join([f"- {issue}" for issue in issues[-5:]])  # Show last 5 issues
        fixes_str = "\n".join([f"- {fix}" for fix in state.get("fixes_implemented", [])])
        
        improvement_request = HumanMessage(
            content=f"""
            Please implement code improvements to fix the following issues:
            
            ISSUES TO FIX:
            {issues_str}
            
            PREVIOUSLY IMPLEMENTED FIXES:
            {fixes_str}
            
            When implementing your solution:
            1. Analyze the root cause of each issue
            2. Develop a targeted fix
            3. Test your implementation in the sandbox
            4. Verify that the fix resolves the issue
            """
        )
        
        # For simplicity, we're not actually modifying code here
        # In a real implementation, this would execute the CodeAct agent
        # and apply changes to the codebase
        
        # Get the compiled CodeAct graph
        code_act_graph = code_agent.compile()
        
        # Invoke the CodeAct agent
        result = code_act_graph.invoke({"messages": [improvement_request]})
        
        # Create a task metric for this code improvement
        metric = TaskMetrics(
            start_time=start_time,
            end_time=time.time(),
            task_id=f"code-improvement-{time.time()}",
            task_description="Code improvement implementation",
            agent_name="code_agent",
            team_name="juno"
        )
        
        # Update metrics
        metrics = list(state.get("metrics", []))
        metrics.append(metric)
        
        # Extract implemented fixes
        response_content = result["messages"][-1].content if "messages" in result else "No changes implemented"
        fixes = []
        if "implemented" in response_content.lower() or "fixed" in response_content.lower():
            for line in response_content.split("\n"):
                if line.strip().startswith("-") or line.strip().startswith("*"):
                    fixes.append(line.strip())
        
        # Update the fixes list
        implemented_fixes = list(state.get("fixes_implemented", []))
        implemented_fixes.extend(fixes)
        
        # Record the code changes
        code_changes = dict(state.get("code_changes", {}))
        change_id = f"change-{time.time()}"
        code_changes[change_id] = {
            "issues_fixed": issues[-5:],
            "implemented_fixes": fixes,
            "timestamp": time.time()
        }
        
        return {
            "messages": [
                HumanMessage(content=response_content, name="code_agent")
            ],
            "next": "supervisor",
            "metrics": metrics,
            "fixes_implemented": implemented_fixes,
            "code_changes": code_changes
        }
    
    # Create the supervisor node
    juno_supervisor_node = make_supervisor_node(
        llm, 
        ["evaluator", "code_agent"],
        """You are the supervisor of the Juno team, responsible for monitoring and improving system performance. 
        
        You have an evaluator agent that tracks metrics and identifies issues, and a code agent that implements improvements. 
        Your job is to coordinate these agents to ensure optimal system performance.
        
        You also oversee resource allocation and scaling. When performance issues are detected due to resource constraints,
        you can direct the code agent to deploy additional resources to teams that need them. You monitor the effectiveness
        of these resource changes and make data-driven decisions about resource allocation.
        
        Additionally, you track missed deadlines and quality issues, implementing improvements to help the system meet its
        targets even as workloads randomly increase. You proactively identify when resources are insufficient and recommend
        scaling up to meet demand.
        """
    )
    
    # Build the Juno team graph
    juno_builder = StateGraph(State)
    juno_builder.add_node("supervisor", juno_supervisor_node)
    juno_builder.add_node("evaluator", evaluator_node)
    juno_builder.add_node("code_agent", code_agent_node)
    
    # Add edges
    juno_builder.add_edge(START, "supervisor")
    juno_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"] or "evaluator",
        {
            "evaluator": "evaluator",
            "code_agent": "code_agent",
            "__end__": "__end__",
        },
    )
    juno_builder.add_edge("evaluator", "supervisor")
    juno_builder.add_edge("code_agent", "supervisor")
    
    return juno_builder.compile()