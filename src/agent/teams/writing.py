"""Writing team implementation for hierarchical agent teams."""

from typing import Literal, Annotated, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, InjectedState

from agent.configuration import Configuration
from agent.state import State
from agent.tools import (
    create_outline, 
    read_document, 
    write_document, 
    edit_document, 
    list_documents
)
from agent.utils import make_supervisor_node


def create_writing_team(config: Configuration) -> StateGraph:
    """Create a writing team with a supervisor, note taker, and document writer."""
    
    # Initialize the language model
    llm = init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
    )
    
    # Create the agents
    note_taking_agent = create_react_agent(
        llm, 
        tools=[create_outline, read_document, list_documents],
        name="note_taker",
        prompt=(
            "You are a note-taking agent that creates outlines based on research. "
            "Your job is to organize information clearly and logically. "
            "Don't ask follow-up questions - work with the available information."
        ),
    )
    
    document_writer_agent = create_react_agent(
        llm, 
        tools=[write_document, edit_document, read_document, list_documents],
        name="doc_writer",
        prompt=(
            "You are a document writing agent that creates well-structured documents. "
            "You can read outlines and turn them into full documents with proper formatting. "
            "Don't ask follow-up questions - use the information you have available."
        ),
    )
    
    # Create note taker node
    def note_taker_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
        """Execute the note taker agent and route back to supervisor."""
        configuration = Configuration.from_runnable_config(config)
        
        # Add working directory to the state for tools
        messages = list(state["messages"])
        if messages and hasattr(messages[-1], "content"):
            updated_content = f"{messages[-1].content}\n\nWorking directory: {configuration.working_directory}"
            messages[-1] = HumanMessage(content=updated_content)
            state_with_working_dir = {**state, "messages": messages}
        else:
            state_with_working_dir = state
        
        result = note_taking_agent.invoke(state_with_working_dir)
        
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="note_taker")
            ],
            "next": "supervisor"
        }
    
    # Create document writer node
    def document_writer_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
        """Execute the document writer agent and route back to supervisor."""
        configuration = Configuration.from_runnable_config(config)
        
        # Add working directory to the state for tools
        messages = list(state["messages"])
        if messages and hasattr(messages[-1], "content"):
            updated_content = f"{messages[-1].content}\n\nWorking directory: {configuration.working_directory}"
            messages[-1] = HumanMessage(content=updated_content)
            state_with_working_dir = {**state, "messages": messages}
        else:
            state_with_working_dir = state
        
        result = document_writer_agent.invoke(state_with_working_dir)
        
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="doc_writer")
            ],
            "next": "supervisor"
        }
    
    # Create the supervisor node
    writing_supervisor_node = make_supervisor_node(
        llm, 
        ["note_taker", "doc_writer"],
        "You are a supervisor managing a document creation team. You have a note-taking agent that creates outlines, and a document writer that creates full documents. Direct them to complete writing tasks efficiently."
    )
    
    # Build the writing team graph
    writing_builder = StateGraph(State)
    writing_builder.add_node("supervisor", writing_supervisor_node)
    writing_builder.add_node("note_taker", note_taker_node)
    writing_builder.add_node("doc_writer", document_writer_node)
    
    # Add edges
    writing_builder.add_edge(START, "supervisor")
    writing_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"] or "note_taker",
        {
            "note_taker": "note_taker",
            "doc_writer": "doc_writer",
            "__end__": "__end__",
        },
    )
    writing_builder.add_edge("note_taker", "supervisor")
    writing_builder.add_edge("doc_writer", "supervisor")
    
    return writing_builder.compile()