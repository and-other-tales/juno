"""Supervisor feedback and grading system for team outputs."""

import time
import random
from typing import Dict, Any, List, Tuple, Optional

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.state import State, AgentPerformance, TaskMetrics
from agent.workload_manager import apply_workload_adjustments


def grade_team_output(
    team_name: str,
    task: str,
    result: str,
    config: Configuration
) -> Tuple[float, str, List[str]]:
    """Grade and provide feedback on a team's output.
    
    Args:
        team_name: The name of the team being graded
        task: The original task description
        result: The result produced by the team
        config: Configuration parameters
        
    Returns:
        Tuple of (score, comments, issues)
    """
    # Initialize the language model
    llm = init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
    )
    
    # Create a prompt for grading the output
    grade_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            f"""You are a supervisor grading the output from the {team_name} team. Your job is to evaluate how well the team performed on the assigned task.
            
            Score on a scale of 0.0 to 1.0, where:
            - 0.0: Completely fails to address the task
            - 0.3: Addresses the task but with major deficiencies
            - 0.5: Adequately addresses the task with some issues
            - 0.7: Well-executed with minor issues
            - 0.9: Excellent execution with tiny improvements possible
            - 1.0: Perfect execution of the task
            
            Your response should be in JSON format with the following fields:
            - score: a float between 0.0 and 1.0
            - comments: detailed feedback for the team
            - issues: list of specific issues or problems identified
            - strengths: list of strengths in the output
            - improvement_suggestions: specific suggestions for improvement
            """
        ),
        (
            "human", 
            """
            TEAM NAME: {team_name}
            
            ORIGINAL TASK:
            {task}
            
            TEAM OUTPUT:
            {result}
            
            Please grade the team's output and provide your evaluation.
            """
        ),
    ])
    
    # Build the grading chain
    grade_chain = grade_prompt | llm
    
    # Invoke the grading chain
    response = grade_chain.invoke({
        "team_name": team_name,
        "task": task,
        "result": result
    })
    
    # Parse the response (in a production system, use a more robust parser)
    try:
        # This is a simple parsing approach - in practice use a better parser
        response_text = response.content
        if "```json" in response_text:
            json_content = response_text.split("```json")[1].split("```")[0]
        elif "```" in response_text:
            json_content = response_text.split("```")[1].split("```")[0]
        else:
            json_content = response_text
            
        import json
        grade_data = json.loads(json_content)
        
        score = float(grade_data.get("score", 0.5))
        comments = grade_data.get("comments", "No comments provided.")
        issues = grade_data.get("issues", [])
        
        return score, comments, issues
        
    except Exception as e:
        # Fallback if parsing fails
        score = 0.5  # Default middle score
        comments = f"Error parsing grade response: {str(e)}. Original response: {response.content}"
        issues = [f"Error in grading: {str(e)}"]
        
        return score, comments, issues


def update_agent_performance(
    state: State,
    team_name: str,
    score: float,
    success: bool = True,
    duration: float = 0.0,
    deadline_met: bool = True
) -> State:
    """Update the performance records for an agent/team.
    
    Args:
        state: The current state
        team_name: The name of the team
        score: The quality score (0-1)
        success: Whether the operation was successful
        duration: The time taken for the operation
        deadline_met: Whether the task was completed within the deadline
        
    Returns:
        Updated state with performance records
    """
    # Get or create the agent performance record
    agent_performances = dict(state.get("agent_performances", {}))
    
    if team_name not in agent_performances:
        agent_performances[team_name] = AgentPerformance(
            agent_id=team_name,
            team_name=team_name
        )
    
    performance = agent_performances[team_name]
    
    # Update the performance record
    performance.quality_scores.append(score)
    if success:
        performance.success_count += 1
    else:
        performance.error_count += 1
    performance.total_time += duration
    
    # Update the team's low quality count if needed
    team_low_quality_counts = dict(state.get("team_low_quality_counts", {}))
    if score < state.get("quality_threshold", 0.7):
        team_low_quality_counts[team_name] = team_low_quality_counts.get(team_name, 0) + 1
    else:
        # Reset the counter on good quality
        team_low_quality_counts[team_name] = 0
    
    # Track missed deadlines
    missed_deadlines_count = state.get("missed_deadlines_count", 0)
    if not deadline_met:
        missed_deadlines_count += 1
    
    # Return the updated state
    return {
        **state,
        "agent_performances": agent_performances,
        "team_low_quality_counts": team_low_quality_counts,
        "missed_deadlines_count": missed_deadlines_count
    }


def create_improvement_request(
    state: State,
    team_name: str,
    issues: List[str],
    resource_request: Optional[Dict[str, Any]] = None
) -> Optional[HumanMessage]:
    """Create an improvement request for the Juno team if needed.
    
    Args:
        state: The current state
        team_name: The team name with issues
        issues: List of issues identified
        resource_request: Optional resource scaling request
        
    Returns:
        A message requesting improvements, or None if not needed
    """
    # Check if we have a resource request
    if resource_request:
        # Format the resource scaling request
        return HumanMessage(
            content=f"""
            RESOURCE SCALING REQUEST

            Team: {resource_request['team']}
            Current agents: {resource_request['current_agents']}
            Recommended agents: {resource_request['recommended_agents']}
            Reason: {resource_request['reason']}
            
            Recent performance issues:
            {chr(10).join([f"- {issue}" for issue in issues])}
            
            Please analyze the current resource allocation and implement the recommended changes.
            After implementation, closely monitor the performance to ensure the changes resolved the issues.
            """,
            name="supervisor"
        )
    
    # Check if the team has a high enough low quality count to trigger improvement
    team_low_quality_counts = state.get("team_low_quality_counts", {})
    low_quality_count = team_low_quality_counts.get(team_name, 0)
    
    # Check if we've missed deadlines recently
    missed_deadlines_count = state.get("missed_deadlines_count", 0)
    
    # Create an improvement request if:
    # 1. The team has consistently low quality OR
    # 2. We've missed multiple deadlines
    if low_quality_count >= 3 or missed_deadlines_count >= 2:
        # Format the improvement request
        all_feedback = state.get("supervisor_feedback", {}).get(team_name, [])
        
        # Determine the primary reason for the improvement request
        reason = ""
        if low_quality_count >= 3:
            reason += f"The {team_name} team has produced low-quality output {low_quality_count} times consecutively."
        
        if missed_deadlines_count >= 2:
            reason += f"\nThe system has missed {missed_deadlines_count} deadlines recently."
        
        return HumanMessage(
            content=f"""
            IMPROVEMENT REQUEST
            
            {reason}
            
            Recent issues:
            {chr(10).join([f"- {issue}" for issue in issues])}
            
            Previous feedback:
            {chr(10).join([f"- {feedback}" for feedback in all_feedback[-5:]])}
            
            Please analyze these issues and implement improvements to enhance the {team_name} team's performance.
            Focus on addressing the recurring problems and improving the overall quality of their output.
            Pay special attention to improving efficiency and meeting deadlines.
            """,
            name="supervisor"
        )
    
    return None


def process_supervisor_feedback(
    state: State, 
    config: RunnableConfig
) -> Dict[str, Any]:
    """Process supervisor feedback for team outputs.
    
    Args:
        state: The current state
        config: Configuration parameters
        
    Returns:
        Updated state with supervisor feedback and routing
    """
    configuration = Configuration.from_runnable_config(config)
    
    # First, apply workload adjustments
    updated_state = apply_workload_adjustments(state, configuration)
    
    # Get the current task
    current_task = updated_state.get("current_task")
    if not current_task:
        return updated_state
    
    # Check if we have team results to grade
    results_to_grade = []
    
    if updated_state.get("research_result"):
        results_to_grade.append(("research", updated_state["research_result"]))
    
    if updated_state.get("writing_result"):
        results_to_grade.append(("writing", updated_state["writing_result"]))
    
    # If no results to grade, return the state with workload adjustments
    if not results_to_grade:
        return updated_state
    
    needs_juno = False
    improvement_message = None
    
    # Get the latest resource change request if any
    resource_request = None
    if updated_state.get("resource_change_requests"):
        resource_request = updated_state["resource_change_requests"][-1]
    
    # Process each team's output
    for team_name, result in results_to_grade:
        # Grade the output
        score, comments, issues = grade_team_output(
            team_name=team_name,
            task=current_task,
            result=result,
            config=configuration
        )
        
        # Check if the deadline was met
        task_deadline = updated_state.get("current_task_deadline", 0)
        current_time = time.time()
        deadline_met = current_time <= task_deadline if task_deadline else True
        
        # Format deadline status
        deadline_status = ""
        if task_deadline:
            if deadline_met:
                time_remaining = task_deadline - current_time
                deadline_status = f"✅ Deadline met with {time_remaining:.1f} seconds remaining."
            else:
                time_overrun = current_time - task_deadline
                deadline_status = f"❌ Deadline missed by {time_overrun:.1f} seconds."
        
        # Create a feedback message
        feedback_message = HumanMessage(
            content=f"""
            Supervisor Feedback for {team_name} team:
            
            Score: {score:.2f}/1.0
            {deadline_status}
            
            Comments: {comments}
            
            Issues:
            {chr(10).join([f"- {issue}" for issue in issues])}
            """,
            name="supervisor"
        )
        
        # Update the state with the feedback
        updated_messages = updated_state.get("messages", []) + [feedback_message]
        updated_state["messages"] = updated_messages
        
        # Update supervisor feedback
        supervisor_feedback = dict(updated_state.get("supervisor_feedback", {}))
        team_feedback = list(supervisor_feedback.get(team_name, []))
        team_feedback.append(comments)
        supervisor_feedback[team_name] = team_feedback
        updated_state["supervisor_feedback"] = supervisor_feedback
        
        # Update performance metrics with deadline status
        updated_state = update_agent_performance(
            updated_state,
            team_name,
            score,
            True,  # Assume success since we got a result
            0.0,   # We don't have the exact duration here
            deadline_met  # Whether the deadline was met
        )
        
        # Check if we need to request improvements (pass resource request if applicable)
        improvement_request = create_improvement_request(
            updated_state,
            team_name,
            issues,
            resource_request
        )
        
        if improvement_request:
            needs_juno = True
            improvement_message = improvement_request
    
    # If we need to route to Juno, update the state
    if needs_juno:
        updated_state["messages"] = updated_state["messages"] + [improvement_message]
        updated_state["next"] = "juno_team"
        
        # Add issues to the identified issues list
        issues_identified = list(updated_state.get("issues_identified", []))
        for team_name, result in results_to_grade:
            _, _, issues = grade_team_output(
                team_name=team_name,
                task=current_task,
                result=result,
                config=configuration
            )
            issues_identified.extend([f"{team_name}: {issue}" for issue in issues])
        
        updated_state["issues_identified"] = issues_identified
    
    # Reset for the next task
    if all(team in [t[0] for t in results_to_grade] for team in ["research", "writing"]):
        # All teams have completed their work, prepare for next task
        updated_state["current_task_deadline"] = 0
        updated_state["current_task_size"] = 1.0
    
    return updated_state