"""Task review and scoring system."""

import time
from typing import Dict, Any, List, Tuple

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig

from agent.configuration import Configuration
from agent.state import State, TaskMetrics


def review_task_result(
    task: str,
    result: str,
    config: Configuration
) -> Tuple[float, str, Dict[str, Any]]:
    """Review and score a task result.
    
    Args:
        task: The original task description
        result: The result produced by the agents
        config: Configuration parameters
        
    Returns:
        Tuple of (score, comments, details)
    """
    # Initialize the language model
    llm = init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
    )
    
    # Create a prompt for reviewing the task
    review_prompt = ChatPromptTemplate.from_messages([
        (
            "system",
            """You are a task review system. Rate the quality of the provided result based on how well it fulfills the original task.
            
            Score on a scale of 0.0 to 1.0, where:
            - 0.0: Completely fails to address the task
            - 0.3: Addresses the task but with major deficiencies
            - 0.5: Adequately addresses the task with some issues
            - 0.7: Well-executed with minor issues
            - 0.9: Excellent execution with tiny improvements possible
            - 1.0: Perfect execution of the task
            
            Your response should be in JSON format with the following fields:
            - score: a float between 0.0 and 1.0
            - comments: detailed feedback on the result
            - strengths: list of strengths in the response
            - weaknesses: list of weaknesses in the response
            - areas_for_improvement: specific suggestions for improvement
            """
        ),
        (
            "human", 
            """
            ORIGINAL TASK:
            {task}
            
            RESULT:
            {result}
            
            Please review the result and provide your evaluation.
            """
        ),
    ])
    
    # Build the review chain
    review_chain = review_prompt | llm
    
    # Invoke the review chain
    response = review_chain.invoke({
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
        review_data = json.loads(json_content)
        
        score = float(review_data.get("score", 0.5))
        comments = review_data.get("comments", "No comments provided.")
        details = {
            "strengths": review_data.get("strengths", []),
            "weaknesses": review_data.get("weaknesses", []),
            "areas_for_improvement": review_data.get("areas_for_improvement", [])
        }
        
        return score, comments, details
        
    except Exception as e:
        # Fallback if parsing fails
        score = 0.5  # Default middle score
        comments = f"Error parsing review response: {str(e)}. Original response: {response.content}"
        details = {"error": str(e)}
        
        return score, comments, details


def update_state_with_review(
    state: State, 
    config: RunnableConfig
) -> Dict[str, Any]:
    """Update the state with a review of the latest task.
    
    Args:
        state: The current state
        config: Configuration parameters
        
    Returns:
        Updated state with review information
    """
    configuration = Configuration.from_runnable_config(config)
    
    # Get the current task and latest result
    current_task = state.get("current_task")
    if not current_task:
        return state
    
    # Get the writing result if it exists (assuming the final output is in writing_result)
    result = state.get("writing_result")
    if not result:
        # If no writing result, try to use the last message
        messages = state.get("messages", [])
        if messages:
            result = messages[-1].content
    
    if not result:
        # Still no result to review
        return state
    
    # Review the task result
    score, comments, details = review_task_result(current_task, result, configuration)
    
    # Update the review scores and comments
    review_scores = dict(state.get("review_scores", {}))
    review_comments = dict(state.get("review_comments", {}))
    
    review_scores[current_task] = score
    review_comments[current_task] = comments
    
    # Create a review message
    review_message = HumanMessage(
        content=f"""
        Task Review Results:
        
        Task: {current_task}
        
        Score: {score:.2f}/1.0
        
        Comments: {comments}
        
        Strengths:
        {chr(10).join([f"- {s}" for s in details.get("strengths", [])])}
        
        Areas for Improvement:
        {chr(10).join([f"- {a}" for a in details.get("areas_for_improvement", [])])}
        """,
        name="reviewer"
    )
    
    # Update metrics with quality score
    metrics = list(state.get("metrics", []))
    for metric in metrics:
        if metric.task_id == current_task:
            metric.response_quality = score
    
    # Update the state with review information
    return {
        **state,
        "messages": state.get("messages", []) + [review_message],
        "review_scores": review_scores,
        "review_comments": review_comments,
        "metrics": metrics
    }