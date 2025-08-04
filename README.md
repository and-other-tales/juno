[![License: Proprietary](https://img.shields.io/badge/license-Proprietary-red.svg)](LICENSE.md)

# Othertales Juno

![Othertales Juno](https://img.shields.io/badge/Othertales-Juno-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)

A self-improving hierarchical multi-agent workflow that autonomously evaluates its own performance, modifies its deployed code, and adapts to changing resource requirements.

## Overview

Othertales Juno is an advanced AI orchestration platform that leverages multiple specialized agent teams operating under a central supervisor. The system's hallmark feature is its ability to evaluate itself, identify weaknesses, and implement code improvements autonomously to its deployed codebase, creating a continuous self-improvement cycle.

### Key Capabilities

- **Autonomous Task Execution**: Generates and completes tasks through specialized teams
- **Performance Evaluation**: Continuously monitors metrics and quality across all operations
- **Self-Modification**: Implements code changes to fix issues and improve performance
- **Quality Control**: Supervisor grades team outputs and triggers improvements when quality issues persist
- **Hierarchical Organization**: Teams work semi-independently under central coordination
- **Dynamic Workload Management**: Supervisor can adjust workload parameters including task volume, complexity, and deadlines
- **Resource Scaling**: Automatically identifies resource constraints and recommends deployment of additional agents
- **Continuous Improvement**: Tests and monitors new resources to optimize overall system performance

## Architecture

```
Top-level Supervisor
│
├── Research Team
│   ├── Research Supervisor
│   ├── Search Agent
│   └── Web Scraper Agent
│
├── Writing Team
│   ├── Writing Supervisor
│   ├── Note Taker Agent
│   └── Document Writer Agent
│
├── Juno Team
│   ├── Juno Supervisor
│   ├── Evaluator Agent
│   └── Code Agent
│
└── Task Generator
```

## Self-Improvement Cycle

Othertales Juno operates on a continuous improvement cycle:

1. **Task Generation**: System creates or receives a task with appropriate deadlines
2. **Resource Assessment**: System evaluates if current resources are sufficient for workload
3. **Task Execution**: Specialized teams (Research, Writing) work on the task within deadlines
4. **Quality Assessment**: Supervisor evaluates and grades team outputs
5. **Performance Monitoring**: System tracks metrics, quality scores, and deadline compliance
6. **Issue Identification**: Issues are identified when:
   - Quality falls below thresholds three times in a row
   - Performance metrics show problems
   - Deadlines are consistently missed
   - Resource constraints are detected
7. **Resource Scaling**: Juno team analyzes resource needs and recommends additional agents when needed
8. **Code Improvement**: The Juno team implements code fixes and resource optimizations
9. **Verification**: Changes are tested to ensure they resolve the identified issues
10. **Deployment**: Approved changes are integrated into the running system
11. **Resource Testing**: New resources are monitored and evaluated for effectiveness
12. **Cycle Restart**: The system begins a new task with improved capabilities and optimized resources

<img src="/img/workflow.png" alt="Workflow Diagram" width="50%">

This creates a genuinely self-improving system that gets better over time and adapts to changing workload demands without human intervention.

## Use Cases

- **Research and Analysis**: Autonomously research topics and create well-structured reports
- **Content Creation**: Generate outlines and documents with automatic quality control
- **System Improvement**: Identify and fix bottlenecks in AI systems
- **Continuous Learning**: Improve performance based on operational experience
- **Autonomous Operation**: Complete sequences of tasks with minimal human oversight
- **Dynamic Scaling**: Automatically adjust resources based on workload demands
- **Deadline Management**: Balance resource allocation to meet critical deadlines
- **Self-Optimization**: Continuously refine its own codebase for better performance

## Installation

### Docker (Recommended)

```bash
# Pull the image
docker pull othertales/juno:latest

# Run the container
docker run -p 8000:8000 \
  -e OPENAI_API_KEY=your_openai_key \
  -e TAVILY_API_KEY=your_tavily_key \
  othertales/juno
```

### Local Installation

1. Clone this repository
2. Install dependencies:

```bash
pip install -e .
```

3. Configure environment variables:

```bash
export OPENAI_API_KEY=your_openai_key
export TAVILY_API_KEY=your_tavily_key
```

## Usage

### LangGraph Server

The easiest way to run Othertales Juno is with LangGraph server:

```bash
othertales-juno serve
```

This starts the server on http://localhost:8000 with UI at http://localhost:8000/ui

### Python API

```python
from othertales_juno import JunoSystem
from langchain_core.messages import HumanMessage

# Initialize the system
juno = JunoSystem()

# Run with a specific task
messages = [HumanMessage(content="Research quantum computing and write a report on its applications")]
result = juno.run(messages)

# Run in autonomous mode for 5 cycles
result = juno.run_cycles(5)

# Configure options with workload and deadline management
juno = JunoSystem(
    model="anthropic/claude-3-7-sonnet",
    max_cycles=10,
    allow_code_changes=True,
    quality_threshold=0.75,
    enable_dynamic_workload=True,
    resource_scaling=True,
    default_deadline_minutes=30,
    min_agents_per_team=1,
    max_agents_per_team=3
)
```

## Configuration

Othertales Juno can be configured through the `config.yaml` file or environment variables:

```yaml
# config.yaml
model_provider: "openai"  # or "anthropic"
model_name: "gpt-4o"
working_directory: "./workspace"
auto_generate_tasks: true
max_cycles: 10
allow_code_changes: true
quality_threshold: 0.7
enable_dynamic_workload: true
resource_scaling: true
default_deadline_minutes: 30
min_agents_per_team: 1
max_agents_per_team: 3
random_workload_increase: 0.3  # 30% chance to increase workload each cycle
max_task_size_multiplier: 2.0  # Maximum task size can be doubled
```

## Security Considerations

Othertales Juno includes a secure sandboxed environment for executing code modifications. However, you should still configure appropriate permissions and review changes in a production environment.

The system's code modification capabilities can be disabled by setting `allow_code_changes: false`.

## Development

For development purposes:

```bash
# Install dev dependencies
pip install -e ".[dev]"

# Run tests
pytest

# Run linting
ruff check .
```

## Testing Dynamic Workload Management

To test the dynamic workload functionality:

1. Initialize the system with workload management enabled:

```python
from othertales_juno import JunoSystem

# Enable dynamic workload with high probability to see it in action
juno = JunoSystem(
    enable_dynamic_workload=True,
    random_workload_increase=0.5,  # 50% chance to increase workload each cycle
    max_task_size_multiplier=2.0,
    default_deadline_minutes=10
)

# Run several cycles to observe workload changes
results = juno.run_cycles(5)
```

2. Monitor the system logs to observe:
   - Random workload increases
   - Deadline setting and tracking
   - Resource scaling recommendations
   - Performance with additional resources

3. You can also manually trigger a workload increase:

```python
from othertales_juno import JunoSystem

juno = JunoSystem()

# Start with a standard task
juno.run(["Research quantum computing basics"])

# Manually increase workload and set tighter deadline
juno.state.current_task_size = 1.8  # 80% larger task
juno.state.current_task_deadline = time.time() + 300  # 5 minute deadline

# Run a more complex task with the tighter deadline
juno.run(["Create a comprehensive report on quantum computing applications in cryptography"])
```

This will allow you to see how the system responds to increased workload, determines when resources are insufficient, and scales accordingly.

## Acknowledgments

Othertales Juno is built using [LangGraph](https://github.com/langchain-ai/langgraph), [LangChain](https://github.com/langchain-ai/langchain), and other open-source technologies.


Copyright © 2025 Adventures of the Persistently Impaired (...and Other Tales) Limited of 85 Great Portland Street, London W1W 7LT
