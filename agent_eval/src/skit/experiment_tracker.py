from sqlmodel import Field, Relationship, Session, SQLModel, create_engine, select
from sqlalchemy import Column, JSON
from datetime import datetime
import hashlib
import json
from sqlalchemy.pool import NullPool

# Capture current timestamp for example data (if uncommented at bottom of file)


class AgentConfigurations(SQLModel, table=True):
    """
    Agent Configuration Database Model.

    Stores different agent configurations for experiment tracking. Each agent configuration
    represents a unique combination of LLM, instructions, and tools. Deduplication is handled
    via hash comparison (not at database level) to allow semantic matching.

    Attributes:
        id: Primary key, auto-generated
        agent_llm: LLM model identifier (e.g., "gpt-4o", "claude-3-5-sonnet")
        agent_name: Human-readable name for tracking (e.g., "bridge_inspector_v1")
        agent_instructions: System prompt/instructions given to the agent
        agent_tools: List of tool names available to the agent (stored as JSON)
        experiments: Relationship to all experiments run with this configuration
    """
    id: int | None = Field(default=None, primary_key=True)
    agent_llm: str  # LLM used to drive the agent
    agent_name: str  # Name of the agent used to track experiments, should be descriptive
    agent_instructions: str  # Instructions given to the agent to test varying prompts
    agent_tools: list[str] = Field(default_factory=list, sa_column=Column(JSON))  # Names of tools given to the agent
    experiments: list["ExperimentResults"] = Relationship(back_populates="agentconfigurations")

class ExperimentResults(SQLModel, table=True):
    """
    Experiment Results Database Model.

    Stores the results of running an agent on a specific question within a scene. Each experiment
    is uniquely identified by the combination of agent configuration and question (enforced via
    unique constraint on agent_and_q_hash).

    Deduplication Strategy:
    - The agent_and_q_hash field contains SHA256(agent_hash + question)
    - Database-level UNIQUE constraint prevents duplicate experiments
    - This ensures you can't accidentally run the same experiment twice

    Attributes:
        id: Primary key, auto-generated
        agentconfiguration_id: Foreign key to AgentConfigurations
        agentconfigurations: Relationship back to the agent configuration used
        experiment_datetime: Timestamp when the experiment was run (stored as string)
        scene_dir: Directory containing the scene data (typically matches PDF filename)
        question: The question posed to the agent
        reference_answer: Ground truth answer from human annotation or oracle
        reference_images: List of image filenames that contain the answer (optional)
        condition_rating: Ground truth condition rating (1-9 scale, optional)
        agent_answer: The agent's response to the question
        agent_condition_rating: Condition rating provided by the agent (optional)
        agent_reference_images: Images referenced by the agent in its answer (optional)
        agent_and_q_hash: SHA256 hash of (agent_config + question) for uniqueness enforcement
    """
    id: int | None = Field(default=None, primary_key=True)
    agentconfiguration_id: int | None = Field(foreign_key="agentconfigurations.id")
    agentconfigurations: AgentConfigurations = Relationship(back_populates="experiments")
    experiment_datetime: str  # Date and time the experiment was run
    scene_dir: str  # Directory containing the scene (typically matches PDF filename)
    question: str  # Question posed to the agent
    reference_answer: str  # Ground truth answer
    reference_images: None | list[str] = Field(default=None, sa_column=Column(JSON))  # Ground truth reference images
    condition_rating: None | int  # Ground truth condition rating (1-9 scale)
    agent_answer: str  # Agent's response
    agent_condition_rating: None | int  # Condition rating provided by agent
    agent_reference_images: None | list[str] = Field(default=None, sa_column=Column(JSON))  # Images cited by agent
    agent_and_q_hash: str = Field(unique=True)  # Unique hash: SHA256(agent_config + question)


# # Database configuration
# sqlite_file_name = "experiments.db"
# sqlite_url = f"sqlite:///{sqlite_file_name}"

# # Initialize database engine and create tables if they don't exist
# engine = create_engine(sqlite_url)
# SQLModel.metadata.create_all(engine)


# =============================================================================
# Hash Functions - Used for deduplication of agents and experiments
# =============================================================================

def get_agent_hash(agent_llm: str, agent_name: str, agent_instructions: str, agent_tools: list[str]) -> str:
    """
    Create a deterministic SHA256 hash from agent configuration parameters.

    This hash is used to identify semantically identical agent configurations, even if they
    are created as separate instances. Tools are sorted before hashing to ensure consistent
    results regardless of input order.

    Args:
        agent_llm: LLM model identifier (e.g., "gpt-4o")
        agent_name: Human-readable agent name (e.g., "bridge_inspector_v1")
        agent_instructions: System prompt/instructions for the agent
        agent_tools: List of tool names available to the agent

    Returns:
        64-character hexadecimal SHA256 hash string

    Example:
        >>> get_agent_hash("gpt-4o", "inspector", "You are helpful", ["map", "oracle"])
        'a3f5e8...'  # 64-char hash
    """
    config_str = json.dumps({
        "llm": agent_llm,
        "name": agent_name,
        "instructions": agent_instructions,
        "tools": sorted(agent_tools)  # Sort to ensure consistent hash
    }, sort_keys=True)
    return hashlib.sha256(config_str.encode()).hexdigest()

def get_experiment_hash(agent_config: AgentConfigurations, question: str) -> str:
    """
    Create a deterministic SHA256 hash from agent configuration + question.

    This hash uniquely identifies an experiment by combining the agent's configuration
    with the specific question being asked. The question is stripped of leading/trailing
    whitespace before hashing to avoid duplicates from formatting differences.

    Args:
        agent_config: The agent configuration object containing LLM, instructions, and tools
        question: The question being asked to the agent

    Returns:
        64-character hexadecimal SHA256 hash string representing the unique experiment

    Example:
        >>> agent = AgentConfigurations(llm="gpt-4o", ...)
        >>> get_experiment_hash(agent, "What is the condition rating?")
        'b2c4d6...'  # 64-char hash
    """
    # Get the agent configuration hash
    agent_hash = get_agent_hash(
        agent_config.agent_llm,
        agent_config.agent_name,
        agent_config.agent_instructions,
        agent_config.agent_tools
    )

    # Combine agent hash with question (stripped to avoid whitespace-only duplicates)
    combined_str = json.dumps({
        "agent_hash": agent_hash,
        "question": question.strip()
    }, sort_keys=True)

    return hashlib.sha256(combined_str.encode()).hexdigest()

# =============================================================================
# Query Functions - Check existence of agents and experiments
# =============================================================================

def agent_exists(session: Session, agent_config: AgentConfigurations) -> AgentConfigurations | None:
    """
    Check if an agent configuration already exists in the database based on hash comparison.

    This function computes the hash of the provided agent_config and compares it against
    all existing agents in the database. If a match is found, it returns the existing
    database record (attached to the current session).

    Note: This performs semantic matching via hash comparison, not database primary key lookup.

    Args:
        session: Active SQLModel database session
        agent_config: Agent configuration to search for (can be an unsaved instance)

    Returns:
        AgentConfigurations object from database if found, None otherwise

    Example:
        >>> with Session(engine) as session:
        ...     config = AgentConfigurations(llm="gpt-4o", name="test", ...)
        ...     existing = agent_exists(session, config)
        ...     if existing:
        ...         print(f"Found existing agent with ID: {existing.id}")
    """
    config_hash = get_agent_hash(
        agent_config.agent_llm,
        agent_config.agent_name,
        agent_config.agent_instructions,
        agent_config.agent_tools
    )

    # Get all agents and check their hashes (semantic matching, not ID-based)
    statement = select(AgentConfigurations)
    agents = session.exec(statement).all()

    for agent in agents:
        existing_hash = get_agent_hash(
            agent.agent_llm,
            agent.agent_name,
            agent.agent_instructions,
            agent.agent_tools
        )
        if existing_hash == config_hash:
            return agent

    return None

def create_agent_if_not_exists(session: Session, agent_config: AgentConfigurations) -> AgentConfigurations:
    """
    Create a new agent configuration if it doesn't already exist.

    Args:
        session: Database session
        agent_config: The agent configuration to create

    Returns:
        AgentConfigurations: agent
    """
    # Check if agent already exists
    existing = agent_exists(session, agent_config)
    if existing:
        return (existing, False)

    # Create new agent
    session.add(agent_config)
    session.commit()
    session.refresh(agent_config)

    return (agent_config, True)

def experiment_exists(session: Session, agent_config: AgentConfigurations, question: str) -> ExperimentResults | None:
    """
    Check if an experiment with this agent config + question already exists in the database.

    This function computes the experiment hash (agent_config + question) and queries the
    database for a matching record using the unique agent_and_q_hash field.

    Args:
        session: Active SQLModel database session
        agent_config: The agent configuration object
        question: The question being asked

    Returns:
        ExperimentResults object from database if found, None otherwise

    Example:
        >>> with Session(engine) as session:
        ...     agent = agent_exists(session, agent_config)
        ...     result = experiment_exists(session, agent, "What is the rating?")
        ...     if result:
        ...         print(f"Experiment already run: {result.agent_answer}")
    """
    experiment_hash = get_experiment_hash(agent_config, question)

    # Query for experiment with this hash (uses unique index for fast lookup)
    statement = select(ExperimentResults).where(ExperimentResults.agent_and_q_hash == experiment_hash)
    return session.exec(statement).first()

def get_experiment_result(session: Session, agent_config: AgentConfigurations, question: str) -> ExperimentResults | None:
    """
    Retrieve the experiment result for a specific agent configuration and question.

    Args:
        session: Database session
        agent_config: The agent configuration
        question: The question to look up

    Returns:
        ExperimentResults if found, None otherwise
    """
    return experiment_exists(session, agent_config, question)

# =============================================================================
# Create Functions - Idempotent creation with deduplication
# =============================================================================

def create_experiment_if_not_exists(
    session: Session,
    agent_config: AgentConfigurations,
    experiment_datetime: str,
    scene_dir: str,
    question: str,
    reference_answer: str,
    agent_answer: str,
    reference_images: list[str] | None = None,
    condition_rating: int | None = None,
    agent_condition_rating: int | None = None,
    agent_reference_images: list[str] | None = None
) -> tuple[ExperimentResults, bool]:
    """
    Create a new experiment if it doesn't already exist (idempotent operation).

    This function checks if an experiment with the same agent configuration and question
    already exists. If it does, it returns the existing record. If not, it creates a new
    one with the provided data and automatically populates the agent_and_q_hash field.

    Args:
        session: Active SQLModel database session
        agent_config: The agent configuration (must have an ID - be saved to database)
        experiment_datetime: Timestamp when experiment was run (as string)
        scene_dir: Directory containing the scene data
        question: The question posed to the agent
        reference_answer: Ground truth answer
        agent_answer: The agent's response
        reference_images: Optional list of ground truth reference image filenames
        condition_rating: Optional ground truth condition rating (1-9 scale)
        agent_condition_rating: Optional agent's condition rating
        agent_reference_images: Optional list of images referenced by agent

    Returns:
        tuple[ExperimentResults, bool]: (experiment, created) where created=True if new experiment was created

    Raises:
        ValueError: If agent_config.id is None (agent must be saved to database first)

    Example:
        >>> with Session(engine) as session:
        ...     agent, _ = create_agent_if_not_exists(session, agent_config)
        ...     exp, created = create_experiment_if_not_exists(
        ...         session, agent, str(datetime.now()), "scene_001",
        ...         "What is the rating?", "Rating is 5", "Agent says 5"
        ...     )
        ...     print(f"{'Created' if created else 'Found existing'} experiment")
    """
    # Check if experiment already exists
    existing = experiment_exists(session, agent_config, question)
    if existing:
        return existing

    # Create new experiment with hash automatically populated
    experiment = ExperimentResults(
        agentconfiguration_id=agent_config.id,
        experiment_datetime=experiment_datetime,
        scene_dir=scene_dir,
        question=question,
        reference_answer=reference_answer,
        reference_images=reference_images,
        condition_rating=condition_rating,
        agent_answer=agent_answer,
        agent_condition_rating=agent_condition_rating,
        agent_reference_images=agent_reference_images,
        agent_and_q_hash=get_experiment_hash(agent_config, question)
    )

    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    return experiment


def create_or_update_experiment(
    session: Session,
    agent_config: AgentConfigurations,
    experiment_datetime: str,
    scene_dir: str,
    question: str,
    reference_answer: str,
    agent_answer: str,
    reference_images: list[str] | None = None,
    condition_rating: int | None = None,
    agent_condition_rating: int | None = None,
    agent_reference_images: list[str] | None = None
) -> tuple[ExperimentResults, bool]:
    """
    Create a new experiment or update an existing one.

    This function checks if an experiment with the same agent configuration and question
    already exists. If it does AND it's a failed experiment (agent_answer starts with "ERROR:"),
    it updates the record with new results. Otherwise, it returns the existing record or creates a new one.

    Args:
        session: Active SQLModel database session
        agent_config: The agent configuration (must have an ID - be saved to database)
        experiment_datetime: Timestamp when experiment was run (as string)
        scene_dir: Directory containing the scene data
        question: The question posed to the agent
        reference_answer: Ground truth answer
        agent_answer: The agent's response
        reference_images: Optional list of ground truth reference image filenames
        condition_rating: Optional ground truth condition rating (1-9 scale)
        agent_condition_rating: Optional agent's condition rating
        agent_reference_images: Optional list of images referenced by agent

    Returns:
        tuple[ExperimentResults, bool]: (experiment, updated) where updated=True if existing experiment was updated

    Example:
        >>> with Session(engine) as session:
        ...     agent, _ = create_agent_if_not_exists(session, agent_config)
        ...     exp, updated = create_or_update_experiment(
        ...         session, agent, str(datetime.now()), "scene_001",
        ...         "What is the rating?", "Rating is 5", "Agent says 5"
        ...     )
        ...     print(f"{'Updated' if updated else 'Created/Found'} experiment")
    """
    # Check if experiment already exists
    existing = experiment_exists(session, agent_config, question)

    if existing:
        # If it's a failed experiment, update it
        if existing.agent_answer and existing.agent_answer.startswith("ERROR:"):
            existing.experiment_datetime = experiment_datetime
            existing.agent_answer = agent_answer
            existing.agent_condition_rating = agent_condition_rating
            existing.agent_reference_images = agent_reference_images
            session.add(existing)
            session.commit()
            session.refresh(existing)
            return (existing, True)
        else:
            # Successful experiment already exists, return it
            return (existing, False)

    # Create new experiment with hash automatically populated
    experiment = ExperimentResults(
        agentconfiguration_id=agent_config.id,
        experiment_datetime=experiment_datetime,
        scene_dir=scene_dir,
        question=question,
        reference_answer=reference_answer,
        reference_images=reference_images,
        condition_rating=condition_rating,
        agent_answer=agent_answer,
        agent_condition_rating=agent_condition_rating,
        agent_reference_images=agent_reference_images,
        agent_and_q_hash=get_experiment_hash(agent_config, question)
    )

    session.add(experiment)
    session.commit()
    session.refresh(experiment)

    return (experiment, False)


# =============================================================================
# ExperimentTracker - High-level API with automatic session management
# =============================================================================

class ExperimentTracker:
    """
    High-level API for experiment tracking.

    Used to cache agent configurations to track experiments and cache QA's from an agent.

    Attributes:
        engine: SQLAlchemy engine for database connections
        db_url: Database connection string

    Example Usage:
        >>> tracker = ExperimentTracker()
        >>>
        >>> # Create agent (no session needed!)
        >>> agent_config = AgentConfigurations(
        ...     agent_llm="gpt-4o",
        ...     agent_name="bridge_inspector_v1",
        ...     agent_instructions="You are a bridge inspector",
        ...     agent_tools=["map_tool", "oracle_tool"]
        ... )
        >>> agent, created = tracker.create_or_get_agent(agent_config)
        >>> print(f"Agent ID: {agent.id}, Created: {created}")
        >>>
        >>> # Create experiment (automatically handles agent re-fetch)
        >>> exp, created = tracker.create_or_get_experiment(
        ...     agent_config=agent_config,
        ...     question="What is the condition rating?",
        ...     experiment_datetime=str(datetime.now()),
        ...     scene_dir="bridge_001",
        ...     reference_answer="Rating is 5",
        ...     agent_answer="I estimate rating of 5",
        ...     condition_rating=5,
        ...     agent_condition_rating=5
        ... )
        >>> print(f"Experiment ID: {exp.id}, Created: {created}")
        >>>
        >>> # Query existing experiment
        >>> result = tracker.get_experiment(agent_config, "What is the condition rating?")
        >>> if result:
        ...     print(f"Found: {result.agent_answer}")
    """

    def __init__(self, db_url: str = "sqlite:///experiments.db"):
        """
        Initialize ExperimentTracker with database connection.

        Args:
            db_url: SQLAlchemy database URL (default: sqlite:///experiments.db)

        Example:
            >>> tracker = ExperimentTracker()  # Uses default SQLite
            >>> tracker = ExperimentTracker("postgresql://user:pass@host/db")  # PostgreSQL
        """
        self.db_url = db_url
        self.engine = create_engine(db_url, poolclass=NullPool)
        SQLModel.metadata.create_all(self.engine)

    def create_or_get_agent(self, agent_config: AgentConfigurations) -> AgentConfigurations:
        """
        Create a new agent configuration or retrieve existing one (idempotent).

        This method manages the database session internally. If an agent with the same
        configuration already exists (based on hash comparison), it returns the existing
        agent. Otherwise, it creates and returns a new one.

        Args:
            agent_config: Agent configuration object (can be unsaved instance)

        Returns:
            AgentConfigurations: The agent configuration (from database, with ID set)

        Example:
            >>> tracker = ExperimentTracker()
            >>> config = AgentConfigurations(
            ...     agent_llm="gpt-4o",
            ...     agent_name="inspector_v1",
            ...     agent_instructions="You are helpful",
            ...     agent_tools=["map", "oracle"]
            ... )
            >>> agent = tracker.create_or_get_agent(config)
            >>> print(f"Agent ID: {agent.id}, Name: {agent.agent_name}")
        """
        with Session(self.engine) as session:
            agent, _ = create_agent_if_not_exists(session, agent_config)
            return agent

    def get_agent(self, agent_config: AgentConfigurations) -> AgentConfigurations | None:
        """
        Retrieve an existing agent configuration from the database.

        Args:
            agent_config: Agent configuration to search for (based on hash matching)

        Returns:
            AgentConfigurations if found, None otherwise

        Example:
            >>> tracker = ExperimentTracker()
            >>> config = AgentConfigurations(llm="gpt-4o", name="test", ...)
            >>> agent = tracker.get_agent(config)
            >>> if agent:
            ...     print(f"Found agent: {agent.agent_name} (ID: {agent.id})")
        """
        with Session(self.engine) as session:
            return agent_exists(session, agent_config)

    def create_or_get_experiment(
        self,
        agent_config: AgentConfigurations,
        question: str,
        experiment_datetime: str,
        scene_dir: str,
        reference_answer: str,
        agent_answer: str,
        reference_images: list[str] | None = None,
        condition_rating: int | None = None,
        agent_condition_rating: int | None = None,
        agent_reference_images: list[str] | None = None
    ) -> ExperimentResults:
        """
        Create a new experiment or retrieve existing one (idempotent).

        This method handles the "detached instance" problem automatically by re-fetching
        the agent configuration within its own session. If an experiment with the same
        agent config + question already exists, it returns the existing experiment.

        Args:
            agent_config: Agent configuration (will be re-fetched from database)
            question: Question posed to the agent
            experiment_datetime: Timestamp when experiment was run (as string)
            scene_dir: Directory containing scene data
            reference_answer: Ground truth answer
            agent_answer: Agent's response
            reference_images: Optional list of ground truth reference images
            condition_rating: Optional ground truth condition rating (1-9)
            agent_condition_rating: Optional agent's condition rating
            agent_reference_images: Optional list of images cited by agent

        Returns:
            ExperimentResults: The experiment result (from database, with ID set)

        Raises:
            ValueError: If agent configuration is not found in database

        Example:
            >>> tracker = ExperimentTracker()
            >>> # First create the agent
            >>> agent_config = AgentConfigurations(...)
            >>> agent = tracker.create_or_get_agent(agent_config)
            >>>
            >>> # Now create experiment (can be in different script/session)
            >>> exp = tracker.create_or_get_experiment(
            ...     agent_config=agent_config,
            ...     question="What is the rating?",
            ...     experiment_datetime=str(datetime.now()),
            ...     scene_dir="bridge_001",
            ...     reference_answer="5",
            ...     agent_answer="5",
            ...     condition_rating=5
            ... )
            >>> print(f"Experiment ID: {exp.id}")
        """
        with Session(self.engine) as session:
            # Re-fetch agent to avoid detached instance error
            agent = agent_exists(session, agent_config)
            if not agent:
                raise ValueError(
                    f"Agent configuration not found in database. "
                    f"Please create agent using create_or_get_agent() first. "
                    f"Agent name: {agent_config.agent_name}"
                )

            return create_experiment_if_not_exists(
                session=session,
                agent_config=agent,
                experiment_datetime=experiment_datetime,
                scene_dir=scene_dir,
                question=question,
                reference_answer=reference_answer,
                agent_answer=agent_answer,
                reference_images=reference_images,
                condition_rating=condition_rating,
                agent_condition_rating=agent_condition_rating,
                agent_reference_images=agent_reference_images
            )

    def get_experiment(
        self,
        agent_config: AgentConfigurations,
        question: str
    ) -> ExperimentResults | None:
        """
        Retrieve an existing experiment result from the database.

        Args:
            agent_config: Agent configuration used in the experiment
            question: Question that was asked

        Returns:
            ExperimentResults if found, None otherwise

        Example:
            >>> tracker = ExperimentTracker()
            >>> agent_config = AgentConfigurations(...)
            >>> result = tracker.get_experiment(agent_config, "What is the rating?")
            >>> if result:
            ...     print(f"Agent answer: {result.agent_answer}")
            ...     print(f"Reference answer: {result.reference_answer}")
        """
        with Session(self.engine) as session:
            # Re-fetch agent to avoid detached instance
            agent = agent_exists(session, agent_config)
            if not agent:
                return None
            return experiment_exists(session, agent, question)

    def get_all_experiments_for_agent(
        self,
        agent_config: AgentConfigurations
    ) -> list[ExperimentResults]:
        """
        Retrieve all experiments run with a specific agent configuration.

        Args:
            agent_config: Agent configuration to query for

        Returns:
            List of ExperimentResults (empty list if none found)

        Example:
            >>> tracker = ExperimentTracker()
            >>> agent_config = AgentConfigurations(...)
            >>> experiments = tracker.get_all_experiments_for_agent(agent_config)
            >>> print(f"Found {len(experiments)} experiments")
            >>> for exp in experiments:
            ...     print(f"Q: {exp.question}, A: {exp.agent_answer}")
        """
        with Session(self.engine) as session:
            agent = agent_exists(session, agent_config)
            if not agent:
                return []

            statement = select(ExperimentResults).where(
                ExperimentResults.agentconfiguration_id == agent.id
            )
            return list(session.exec(statement).all())

    def create_or_update_experiment(
        self,
        agent_config: AgentConfigurations,
        question: str,
        experiment_datetime: str,
        scene_dir: str,
        reference_answer: str,
        agent_answer: str,
        reference_images: list[str] | None = None,
        condition_rating: int | None = None,
        agent_condition_rating: int | None = None,
        agent_reference_images: list[str] | None = None
    ) -> tuple[ExperimentResults, bool]:
        """
        Create a new experiment or update an existing failed experiment.

        This method handles the "detached instance" problem automatically by re-fetching
        the agent configuration within its own session. If an experiment with the same
        agent config + question already exists and it's a failed experiment, it updates
        it with the new results. Otherwise, it returns the existing experiment or creates a new one.

        Args:
            agent_config: Agent configuration (will be re-fetched from database)
            question: Question posed to the agent
            experiment_datetime: Timestamp when experiment was run (as string)
            scene_dir: Directory containing scene data
            reference_answer: Ground truth answer
            agent_answer: Agent's response
            reference_images: Optional list of ground truth reference images
            condition_rating: Optional ground truth condition rating (1-9)
            agent_condition_rating: Optional agent's condition rating
            agent_reference_images: Optional list of images cited by agent

        Returns:
            tuple[ExperimentResults, bool]: (experiment, updated) where updated=True if failed experiment was updated

        Raises:
            ValueError: If agent configuration is not found in database

        Example:
            >>> tracker = ExperimentTracker()
            >>> agent_config = AgentConfigurations(...)
            >>> agent = tracker.create_or_get_agent(agent_config)
            >>> exp, updated = tracker.create_or_update_experiment(
            ...     agent_config=agent_config,
            ...     question="What is the rating?",
            ...     experiment_datetime=str(datetime.now()),
            ...     scene_dir="bridge_001",
            ...     reference_answer="5",
            ...     agent_answer="5",
            ...     condition_rating=5
            ... )
            >>> print(f"Experiment {'updated' if updated else 'created/found'}")
        """
        with Session(self.engine) as session:
            # Re-fetch agent to avoid detached instance error
            agent = agent_exists(session, agent_config)
            if not agent:
                raise ValueError(
                    f"Agent configuration not found in database. "
                    f"Please create agent using create_or_get_agent() first. "
                    f"Agent name: {agent_config.agent_name}"
                )

            return create_or_update_experiment(
                session=session,
                agent_config=agent,
                experiment_datetime=experiment_datetime,
                scene_dir=scene_dir,
                question=question,
                reference_answer=reference_answer,
                agent_answer=agent_answer,
                reference_images=reference_images,
                condition_rating=condition_rating,
                agent_condition_rating=agent_condition_rating,
                agent_reference_images=agent_reference_images
            )

    def get_all_agents(self) -> list[AgentConfigurations]:
        """
        Retrieve all agent configurations from the database.

        Returns:
            List of all AgentConfigurations (empty list if none exist)

        Example:
            >>> tracker = ExperimentTracker()
            >>> agents = tracker.get_all_agents()
            >>> for agent in agents:
            ...     print(f"{agent.agent_name}: {agent.agent_llm}")
        """
        with Session(self.engine) as session:
            statement = select(AgentConfigurations)
            return list(session.exec(statement).all())
