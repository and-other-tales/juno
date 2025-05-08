"""Define the tools for the hierarchical agent teams."""

import os
from pathlib import Path
from typing import Annotated, List, Dict, Optional, Any

from langchain_community.document_loaders import WebBaseLoader
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.tools import tool

# Create workspace directory for file operations
def get_workspace_path(working_dir: str) -> Path:
    """Get the workspace path, creating it if it doesn't exist."""
    workspace = Path(working_dir)
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace

# Research team tools
tavily_tool = TavilySearchResults(max_results=5)

@tool
def search_web(query: str) -> str:
    """Search the web for information on a given query."""
    return tavily_tool.invoke(query)

@tool
def scrape_webpages(urls: List[str]) -> str:
    """Use requests and bs4 to scrape the provided web pages for detailed information."""
    loader = WebBaseLoader(urls)
    docs = loader.load()
    return "\n\n".join(
        [
            f'<Document name="{doc.metadata.get("title", "")}">\n{doc.page_content}\n</Document>'
            for doc in docs
        ]
    )

# Document writing team tools
@tool
def create_outline(
    points: Annotated[List[str], "List of main points or sections."],
    file_name: Annotated[str, "File path to save the outline."],
    working_dir: Annotated[str, "Working directory path."] = "/tmp/hierarchical_agents_workspace",
) -> Annotated[str, "Path of the saved outline file."]:
    """Create and save an outline."""
    workspace = get_workspace_path(working_dir)
    with (workspace / file_name).open("w") as file:
        for i, point in enumerate(points):
            file.write(f"{i + 1}. {point}\n")
    return f"Outline saved to {file_name}"

@tool
def read_document(
    file_name: Annotated[str, "File path to read the document from."],
    working_dir: Annotated[str, "Working directory path."] = "/tmp/hierarchical_agents_workspace",
    start: Annotated[Optional[int], "The start line. Default is 0"] = None,
    end: Annotated[Optional[int], "The end line. Default is None"] = None,
) -> str:
    """Read the specified document."""
    workspace = get_workspace_path(working_dir)
    file_path = workspace / file_name
    
    if not file_path.exists():
        return f"Error: File {file_name} not found"
    
    with file_path.open("r") as file:
        lines = file.readlines()
    
    if start is None:
        start = 0
    
    return "".join(lines[start:end])

@tool
def write_document(
    content: Annotated[str, "Text content to be written into the document."],
    file_name: Annotated[str, "File path to save the document."],
    working_dir: Annotated[str, "Working directory path."] = "/tmp/hierarchical_agents_workspace",
) -> Annotated[str, "Path of the saved document file."]:
    """Create and save a text document."""
    workspace = get_workspace_path(working_dir)
    with (workspace / file_name).open("w") as file:
        file.write(content)
    return f"Document saved to {file_name}"

@tool
def edit_document(
    file_name: Annotated[str, "Path of the document to be edited."],
    inserts: Annotated[
        Dict[int, str],
        "Dictionary where key is the line number (1-indexed) and value is the text to be inserted at that line.",
    ],
    working_dir: Annotated[str, "Working directory path."] = "/tmp/hierarchical_agents_workspace",
) -> Annotated[str, "Path of the edited document file."]:
    """Edit a document by inserting text at specific line numbers."""
    workspace = get_workspace_path(working_dir)
    file_path = workspace / file_name
    
    if not file_path.exists():
        return f"Error: File {file_name} not found"

    with file_path.open("r") as file:
        lines = file.readlines()

    sorted_inserts = sorted(inserts.items())

    for line_number, text in sorted_inserts:
        if 1 <= line_number <= len(lines) + 1:
            lines.insert(line_number - 1, text + "\n")
        else:
            return f"Error: Line number {line_number} is out of range."

    with file_path.open("w") as file:
        file.writelines(lines)

    return f"Document edited and saved to {file_name}"

@tool
def list_documents(
    working_dir: Annotated[str, "Working directory path."] = "/tmp/hierarchical_agents_workspace",
) -> List[str]:
    """List all documents in the workspace."""
    workspace = get_workspace_path(working_dir)
    return [f.name for f in workspace.iterdir() if f.is_file()]