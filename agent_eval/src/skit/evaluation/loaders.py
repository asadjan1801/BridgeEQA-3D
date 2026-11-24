"""
Database loading utilities for experiment results.

Provides functions to load experiment results from ExperimentTracker databases.
"""

from pathlib import Path
from typing import Any, Dict, List

from sqlmodel import Session, create_engine, select

from skit.experiment_tracker import AgentConfigurations, ExperimentResults


def load_experiments_from_db(db_path: Path | str) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load all experiments from a single database file.

    Args:
        db_path: Path to the experiments.db SQLite database

    Returns:
        Dictionary mapping agent names to lists of experiment results
        Format: {"agent_name": [{"scene_dir": ..., "question": ..., ...}, ...]}

    Raises:
        FileNotFoundError: If database file does not exist

    Example:
        >>> results = load_experiments_from_db("experiments.db")
        >>> print(f"Found {len(results)} agents")
        >>> for agent_name, experiments in results.items():
        ...     print(f"{agent_name}: {len(experiments)} experiments")
    """
    db_path = Path(db_path)
    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    print(f"Loading experiments from: {db_path}")

    # Create engine and session
    engine = create_engine(f"sqlite:///{db_path}")
    agent_results: Dict[str, List[Dict[str, Any]]] = {}

    with Session(engine) as session:
        # Get all agent configurations
        agents = session.exec(select(AgentConfigurations)).all()
        print(f"Found {len(agents)} agent configurations")

        for agent in agents:
            agent_name = agent.agent_name

            # Get all experiments for this agent
            statement = select(ExperimentResults).where(
                ExperimentResults.agentconfiguration_id == agent.id
            )
            experiments = session.exec(statement).all()

            print(f"  - {agent_name}: {len(experiments)} experiments")

            # Convert to dictionary format
            agent_results[agent_name] = []
            for exp in experiments:
                # Skip failed experiments (ERROR: prefix)
                if exp.agent_answer and exp.agent_answer.startswith("ERROR:"):
                    continue

                agent_results[agent_name].append({
                    "scene_dir": exp.scene_dir,
                    "question": exp.question,
                    "reference_answer": exp.reference_answer,
                    "reference_condition_rating": exp.condition_rating,
                    "reference_images": exp.reference_images or [],
                    "agent_answer": exp.agent_answer,
                    "agent_condition_rating": exp.agent_condition_rating,
                    "agent_reference_images": exp.agent_reference_images or [],
                    "error": None,
                    "cached": False,  # Not relevant for analysis
                })

    return agent_results


def load_experiments_from_multiple_dbs(
    db_paths: List[Path | str],
    model_names: List[str] | None = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load and combine experiments from multiple database files.

    Useful for comparing the same agent architectures across different models.

    Args:
        db_paths: List of paths to experiments.db files
        model_names: Optional list of model identifiers to add to agent names
                     (e.g., ["grok-4-fast", "gemini-2.5-flash"])
                     If not provided, will use db0, db1, db2, etc. as suffixes

    Returns:
        Combined dictionary mapping agent names to experiment results
        If model_names provided, agent names will be suffixed with model name

    Example:
        >>> db_paths = ["experiments_grok.db", "experiments_gemini.db"]
        >>> model_names = ["grok-4-fast", "gemini-2.5-flash"]
        >>> results = load_experiments_from_multiple_dbs(db_paths, model_names)
        >>> print(list(results.keys()))
        ['embodied_sg_oracle_both_grok-4-fast', 'embodied_sg_oracle_both_gemini-2.5-flash', ...]
    """
    combined_results: Dict[str, List[Dict[str, Any]]] = {}

    for idx, db_path in enumerate(db_paths):
        suffix = f"_{model_names[idx]}" if model_names else f"_db{idx}"

        results = load_experiments_from_db(db_path)

        # Add to combined results with suffix
        for agent_name, experiments in results.items():
            combined_agent_name = f"{agent_name}{suffix}"
            combined_results[combined_agent_name] = experiments

    return combined_results
