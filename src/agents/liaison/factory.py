"""
Liaison Agent Factory
Dependency injection for creating LiaisonAgent instances.
"""

from __future__ import annotations

import logging
from typing import Optional

from .config import LiaisonConfig
from .liaison import LiaisonAgent
from src.routing.router import get_router, SwitchyardRouter
from src.governance.guardrails import get_guardrails_engine, GuardrailsEngine
from src.vectorstore.faiss_store import FAISSVectorStore, FAISSVectorStoreConfig
from src.state.event_store import get_event_store, EventStore

logger = logging.getLogger(__name__)


async def create_liaison_agent(config: Optional[LiaisonConfig] = None) -> LiaisonAgent:
    """Factory function with full dependency injection."""
    config = config or LiaisonConfig()

    # Router (uses global singleton or creates new)
    router = get_router(config.reasoning.router_config_path)

    # Guardrails (only if config provided)
    guardrails = None
    if config.reasoning.guardrails_config:
        guardrails = get_guardrails_engine(config.reasoning.guardrails_config)

    # Vector Store (if path exists)
    vector_store = None
    try:
        from langchain_community.embeddings import HuggingFaceEmbeddings
        embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
        vector_store = FAISSVectorStore(FAISSVectorStoreConfig(
            path=config.reasoning.vector_store_path,
            embedder=embedder,
        ))
    except Exception as e:
        logger.warning(f"Vector store not available: {e}")

    # Event Store
    event_store = await get_event_store(config.event_store_path)

    # Create agent (session_id will be auto-generated/loaded by agent)
    agent = LiaisonAgent(
        config=config,
        router=router,
        guardrails=guardrails,
        vector_store=vector_store,
        event_store=event_store,
    )

    await agent.initialize()
    return agent


def create_liaison_agent_sync(config: Optional[LiaisonConfig] = None) -> LiaisonAgent:
    """Synchronous factory for non-async contexts."""
    import asyncio
    return asyncio.run(create_liaison_agent(config))
