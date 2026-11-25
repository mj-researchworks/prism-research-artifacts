"""Allowed agent/tool registry for the bounded autonomous workflow."""

from __future__ import annotations

from .agents.critic_verification_agent import CriticVerificationAgent
from .agents.intent_agent import IntentEvidenceAgent
from .agents.preference_reflection_agent import PreferenceReflectionAgent
from .agents.recommendation_decision_agent import RecommendationDecisionAgent
from .agents.search_research_agent import SearchResearchAgent
from .tools.conserve_search_tool import ConServeSearchTool
from .tools.response_tool import ResponseTool


def build_registry() -> dict[str, object]:
    return {
        "intent_agent": IntentEvidenceAgent(),
        "preference_reflection_agent": PreferenceReflectionAgent(),
        "search_research_agent": SearchResearchAgent(),
        "recommendation_decision_agent": RecommendationDecisionAgent(),
        "critic_verification_agent": CriticVerificationAgent(),
        "conserve_search_tool": ConServeSearchTool(),
        "response_tool": ResponseTool(),
    }

