"""Define the configurable parameters for the agent."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from typing import Optional, List, Literal, Dict, Any

from langchain_core.runnables import RunnableConfig


@dataclass(kw_only=True)
class Configuration:
    """The configuration for the hierarchical agent teams."""

    # Model providers and names
    model_provider: str = "openai"
    model_name: str = "gpt-4o"
    
    # API keys (these would be better handled with environment variables)
    openai_api_key: Optional[str] = None
    tavily_api_key: Optional[str] = None
    
    # Team configuration
    enabled_teams: List[Literal["research", "writing", "juno"]] = field(
        default_factory=lambda: ["research", "writing", "juno"]
    )
    
    # Document workspace settings
    working_directory: str = "/tmp/hierarchical_agents_workspace"
    
    # Advanced configuration
    recursion_limit: int = 100
    max_iterations: int = 10
    
    # Debugging
    debug_mode: bool = False
    
    # Auto-run settings
    auto_generate_tasks: bool = True
    task_categories: List[str] = field(
        default_factory=lambda: [
            "Research and report", 
            "Market analysis", 
            "Technical documentation",
            "Creative writing",
            "Data analysis",
            "Summarization"
        ]
    )
    
    # Performance targets
    performance_targets: Dict[str, float] = field(
        default_factory=lambda: {
            "avg_response_time": 10.0,       # seconds
            "success_rate": 0.95,            # 95% success rate
            "response_quality": 0.8,         # quality score (0-1)
            "task_completion_rate": 0.9      # completion rate (0-1)
        }
    )
    
    # Workload management settings
    enable_dynamic_workload: bool = True     # Enable random workload increases
    random_workload_increase: float = 0.3    # 30% chance to increase workload each cycle
    max_task_size_multiplier: float = 2.0    # Maximum task size can be doubled
    default_deadline_minutes: int = 30       # Default deadline in minutes
    
    # Resource scaling settings
    resource_scaling: bool = True            # Whether to enable resource scaling
    min_agents_per_team: int = 1             # Minimum number of agents per team
    max_agents_per_team: int = 3             # Maximum number of agents per team
    
    # Code improvement settings
    allow_code_changes: bool = True
    max_code_changes_per_cycle: int = 3
    code_change_cooldown: int = 2            # cycles to wait between code changes
    
    # Juno team configuration
    juno_evaluation_frequency: int = 1       # evaluate after every N tasks
    code_improvement_threshold: float = 0.7  # trigger improvements when performance < threshold
    
    # Sandbox configuration for code agent
    sandbox_directory: str = "/tmp/hierarchical_agents_sandbox"
    
    # Cycle control
    max_cycles: int = 10                     # maximum number of autonomous cycles

    @classmethod
    def from_runnable_config(
        cls, config: Optional[RunnableConfig] = None
    ) -> Configuration:
        """Create a Configuration instance from a RunnableConfig object."""
        configurable = (config.get("configurable") or {}) if config else {}
        _fields = {f.name for f in fields(cls) if f.init}
        return cls(**{k: v for k, v in configurable.items() if k in _fields})