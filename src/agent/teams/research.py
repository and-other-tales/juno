# Copyright © 2025 PI & Other Tales Inc.. All Rights Reserved.
"""Research team implementation for hierarchical agent teams."""

from typing import Literal, Annotated, Dict, Any

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph import StateGraph, START
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command, InjectedState

from agent.configuration import Configuration
from agent.state import State
from agent.tools import search_web, scrape_webpages
from agent.utils import make_supervisor_node


def create_research_team(config: Configuration) -> StateGraph:
    """Create a research team with a supervisor, search agent, and web scraper agent."""
    
    # Initialize the language model
    llm = init_chat_model(
        config.model_name,
        model_provider=config.model_provider,
    )
    
    # Create the agents
    search_agent = create_react_agent(
        llm, 
        tools=[search_web],
        name="search",
    )
    
    web_scraper_agent = create_react_agent(
        llm, 
        tools=[scrape_webpages],
        name="web_scraper",
    )
    
    # Create search node
    def search_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
        """Execute the search agent and route back to supervisor."""
        result = search_agent.invoke(state)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="search")
            ],
            "next": "supervisor"
        }
    
    # Create web scraper node
    def web_scraper_node(state: Annotated[State, InjectedState], config: RunnableConfig) -> Dict[str, Any]:
        """Execute the web scraper agent and route back to supervisor."""
        result = web_scraper_agent.invoke(state)
        return {
            "messages": [
                HumanMessage(content=result["messages"][-1].content, name="web_scraper")
            ],
            "next": "supervisor"
        }
    
    # Create the supervisor node
    research_supervisor_node = make_supervisor_node(
        llm, 
        ["search", "web_scraper"],
        "You are a supervisor managing a research team. You have a search agent that can find information online, and a web scraper agent that can extract detailed information from web pages. Assign tasks to them to fulfill the research needs."
    )
    
    # Build the research team graph
    research_builder = StateGraph(State)
    research_builder.add_node("supervisor", research_supervisor_node)
    research_builder.add_node("search", search_node)
    research_builder.add_node("web_scraper", web_scraper_node)
    
    # Add edges
    research_builder.add_edge(START, "supervisor")
    research_builder.add_conditional_edges(
        "supervisor",
        lambda state: state["next"] or "search",
        {
            "search": "search",
            "web_scraper": "web_scraper",
            "__end__": "__end__",
        },
    )
    research_builder.add_edge("search", "supervisor")
    research_builder.add_edge("web_scraper", "supervisor")
    
    return research_builder.compile()