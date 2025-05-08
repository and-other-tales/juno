"""Utilities for hierarchical agent teams."""

from typing import List, Literal, Dict, Any, Optional, Callable, cast

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableConfig
from langgraph.graph import END
from langgraph.types import Command, TypedDict


def make_supervisor_node(
    llm: BaseChatModel, 
    members: List[str],
    system_prompt: Optional[str] = None
) -> Callable:
    """Create a supervisor node function.
    
    Args:
        llm: The language model to use
        members: List of member nodes this supervisor can delegate to
        system_prompt: Optional custom system prompt for the supervisor
        
    Returns:
        A node function that can be added to a LangGraph
    """
    options = ["__end__"] + members
    
    if system_prompt is None:
        system_prompt = (
            "You are a supervisor tasked with managing a conversation between the"
            f" following workers: {members}. Given the following request,"
            " respond with the worker to delegate to next. Each worker will perform a"
            " task and respond with their results. When finished,"
            " respond with __end__."
        )
    
    # Create a structured output schema
    class Router(TypedDict):
        """Worker to route to next."""
        next: Literal[tuple(options)]  # type: ignore
    
    # Create the supervisor prompt
    supervisor_prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}"),
    ])
    
    # Create the supervisor chain
    supervisor_chain = supervisor_prompt | llm.with_structured_output(Router)
    
    def supervisor_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
        """Node that routes to the next worker based on the LLM's decision."""
        # Get the human message from the state
        messages = state.get("messages", [])
        
        # Build the input for the supervisor
        full_history = "\n\n".join([
            f"{msg.type.upper()} {getattr(msg, 'name', '')}: {msg.content}" 
            for msg in messages
        ])
        
        # Get the routing decision
        response = supervisor_chain.invoke({"input": full_history})
        goto = response.get("next", "__end__")
        
        # Update the state
        return {"next": goto}
    
    return supervisor_node


def process_team_output(state: Dict[str, Any], team_result: Dict[str, Any]) -> Dict[str, Any]:
    """Process the output from a team subgraph.
    
    Args:
        state: The current state
        team_result: The result from the team subgraph
        
    Returns:
        Updated state
    """
    # Extract the most recent message from the team result
    team_messages = team_result.get("messages", [])
    if not team_messages:
        return state
    
    latest_message = team_messages[-1]
    
    # Create a new state with the updated messages
    messages = state.get("messages", [])
    messages.append(latest_message)
    
    return {
        **state,
        "messages": messages,
    }