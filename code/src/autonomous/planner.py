"""Executive Planner for the first bounded autonomous workflow."""

from __future__ import annotations

from .flags import combined_flags
from .schemas import PlannerDecision


class RuleBasedPlanner:
    """Small deterministic planner skeleton.

    This is intentionally simple for milestone 1. It proves that the system
    path is planner-selected. Later, this class can call an LLM that returns
    the same PlannerDecision schema.
    """

    def decide(self, state: dict) -> PlannerDecision:
        if not state["intent"]["label"]:
            return PlannerDecision(
                action="CALL_AGENT",
                target="intent_agent",
                reason="Need intent evidence before deciding whether to search, ask, or stop.",
            )

        if state["intent"].get("label") == "accept" and state.get("previous_recommendations") and self._has_observation(state, "preference_reflection_agent"):
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="User accepted the current recommendation context; do not search again.",
            )

        if not self._has_observation(state, "preference_reflection_agent"):
            return PlannerDecision(
                action="CALL_AGENT",
                target="preference_reflection_agent",
                reason="Extract or revise preference evidence for the latest user turn.",
            )

        if "underspecified_product_type" in combined_flags(state) and not self._has_product_type(state):
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Product/category is missing after preference reflection; ask the user before building a final search request.",
            )

        if not state["preferences"]:
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="No usable shopping preferences were found after specialist review.",
            )

        if not state.get("final_search_request"):
            return PlannerDecision(
                action="CALL_AGENT",
                target="search_research_agent",
                reason="Build structured search request from current shopping memory.",
            )

        if state.get("final_search_request") and not state.get("raw_candidates") and not state.get("critic_decision"):
            return PlannerDecision(
                action="CALL_AGENT",
                target="critic_verification_agent",
                reason="Structured request exists; critic must decide whether it is ready for retrieval.",
            )

        if state.get("critic_decision") == "ASK_USER" and not state.get("raw_candidates"):
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Critic requires clarification before retrieval.",
            )

        if state.get("critic_next_action") == "update_state" and state.get("final_search_request") and not state.get("raw_candidates"):
            return PlannerDecision(
                action="CALL_AGENT",
                target="search_research_agent",
                reason="Critic requested a state update; rebuild the structured search request and retrieval plan.",
            )

        if state.get("critic_next_action") == "search_again" and state.get("final_search_request"):
            if int(state.get("search_attempts") or 0) >= 2:
                return PlannerDecision(
                    action="CALL_TOOL",
                    target="response_tool",
                    reason="Search-again limit reached; ask the user for one more preference.",
                )
            return PlannerDecision(
                action="CALL_AGENT",
                target="search_research_agent",
                reason="Critic requested another search; broaden the retrieval plan before retrieving again.",
            )

        if (
            state.get("final_search_request")
            and state.get("critic_decision") != "ACCEPT"
            and not state.get("raw_candidates")
        ):
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Structured request is not accepted by critic; respond instead of retrieving.",
            )

        if not state.get("raw_candidates"):
            return PlannerDecision(
                action="CALL_TOOL",
                target="conserve_search_tool",
                reason="Search request is ready; retrieve grounded catalog candidates.",
            )

        if not self._has_post_retrieval_critic_observation(state):
            return PlannerDecision(
                action="CALL_AGENT",
                target="critic_verification_agent",
                reason="Retrieved candidates must be challenged before recommendation.",
            )

        if state.get("critic_decision") == "ASK_USER":
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Critic decision is ASK_USER; planner should not recommend yet.",
            )

        if state.get("critic_decision") == "ACCEPT" and state.get("verified_candidates") and not state.get("last_recommendations"):
            return PlannerDecision(
                action="CALL_AGENT",
                target="recommendation_decision_agent",
                reason="Critic accepted verified candidates; select recommendations for response.",
            )

        return PlannerDecision(
            action="CALL_TOOL",
            target="response_tool",
            reason="Recommendation state is ready; generate final response.",
        )

    def _needs_reflection(self, state: dict) -> bool:
        text = state.get("latest_user_message", "").lower()
        signals = ("actually", "instead", "don't like", "do not like", "not ", "change", "cheaper")
        return any(signal in text for signal in signals)

    def _has_observation(self, state: dict, name: str) -> bool:
        return any(obs.get("component") == name for obs in state.get("tool_observations", []))

    def _has_product_type(self, state: dict) -> bool:
        product_type = (state.get("preferences") or {}).get("product_type")
        if isinstance(product_type, dict):
            return bool(product_type.get("value"))
        return bool(product_type)

    def _has_post_retrieval_critic_observation(self, state: dict) -> bool:
        if not state.get("raw_candidates"):
            return False
        observations = state.get("tool_observations", [])
        search_index = -1
        for index, obs in enumerate(observations):
            if obs.get("component") == "conserve_search_tool":
                search_index = index
        return any(
            obs.get("component") == "critic_verification_agent"
            for obs in observations[search_index + 1 :]
        )



ExecutivePlanner = RuleBasedPlanner
