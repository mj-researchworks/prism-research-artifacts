"""State-aware plan validation for bounded autonomous execution."""

from __future__ import annotations

from .flags import combined_flags
from .schemas import PlannerDecision


AGENTS = {
    "intent_agent",
    "preference_reflection_agent",
    "search_research_agent",
    "critic_verification_agent",
    "recommendation_decision_agent",
}
TOOLS = {"conserve_search_tool", "response_tool"}
ALLOWED_TARGETS = AGENTS | TOOLS


class PlanPolicy:
    """Validates planner proposals against the current blackboard state.

    The LLM planner may propose a useful high-level route, but state can change
    after each specialist runs. This policy keeps autonomy bounded by enforcing
    required evidence, no unnecessary retrieval on acceptance, and critic review
    before recommendation.
    """

    def override(self, state: dict, queued_plan: list[str] | None = None) -> PlannerDecision | None:
        if state.get("intent", {}).get("label") == "accept" and state.get("previous_recommendations"):
            if not self.has_observation(state, "preference_reflection_agent"):
                return PlannerDecision(
                    action="CALL_AGENT",
                    target="preference_reflection_agent",
                    reason="Plan policy: acceptance needs preference reflection before final response.",
                )
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Plan policy: user accepted current recommendations, so skip retrieval.",
            )

        if state.get("intent", {}).get("label") == "accept" and not state.get("previous_recommendations"):
            if not self.has_observation(state, "preference_reflection_agent"):
                return PlannerDecision(
                    action="CALL_AGENT",
                    target="preference_reflection_agent",
                    reason="Plan policy: acceptance must update memory before responding.",
                )
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Plan policy: user accepted, but there are no recommendations in memory to accept.",
            )

        pending = queued_plan or []

        flags = set(combined_flags(state))
        pre_retrieval_flags = {"underspecified_product_type", "missing_search_request", "ask_user_before_retrieval"}
        if flags.intersection(pre_retrieval_flags) and not state.get("final_search_request") and not state.get("raw_candidates"):
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Plan policy: required search information is missing, so ask the user before retrieval.",
            )

        if (
            state.get("final_search_request")
            and flags.intersection(pre_retrieval_flags)
            and not state.get("raw_candidates")
            and not state.get("critic_decision")
        ):
            if pending and pending[0] == "critic_verification_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="critic_verification_agent",
                reason="Plan policy: search request needs clarification/safety review before retrieval.",
            )

        if state.get("critic_decision") == "ASK_USER" and not state.get("raw_candidates"):
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Plan policy: critic requires clarification before retrieval.",
            )

        if state.get("critic_next_action") == "update_state" and state.get("final_search_request") and not state.get("raw_candidates"):
            if pending and pending[0] == "search_research_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="search_research_agent",
                reason="Plan policy: critic requested state update; rebuild search request and retrieval plan.",
            )

        if state.get("critic_next_action") == "search_again" and state.get("final_search_request"):
            if int(state.get("search_attempts") or 0) >= 2:
                return PlannerDecision(
                    action="CALL_TOOL",
                    target="response_tool",
                    reason="Plan policy: search-again limit reached; ask user for one more preference.",
                )
            if pending and pending[0] == "search_research_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="search_research_agent",
                reason="Plan policy: critic requested another search; ask Search Agent to broaden the retrieval plan.",
            )

        if (
            self.has_product_type(state)
            and self.has_observation(state, "preference_reflection_agent")
            and not state.get("final_search_request")
            and not state.get("raw_candidates")
        ):
            if pending and pending[0] == "search_research_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="search_research_agent",
                reason="Plan policy: product/category is known, so build the structured search request before responding.",
            )

        if not state.get("final_search_request") and pending and pending[0] in {"conserve_search_tool", "recommendation_decision_agent"}:
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Plan policy: retrieval/recommendation cannot run before a structured search request exists.",
            )

        if (
            state.get("final_search_request")
            and not state.get("raw_candidates")
            and not state.get("critic_decision")
        ):
            if pending and pending[0] == "critic_verification_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="critic_verification_agent",
                reason="Plan policy: structured request requires pre-retrieval critic readiness check.",
            )

        if (
            state.get("final_search_request")
            and state.get("critic_decision") == "ACCEPT"
            and not state.get("raw_candidates")
        ):
            if pending and pending[0] == "conserve_search_tool":
                return None
            return PlannerDecision(
                action="CALL_TOOL",
                target="conserve_search_tool",
                reason="Plan policy: structured search request is ready, so retrieve candidates before responding.",
            )

        if state.get("raw_candidates") and not self.has_post_retrieval_critic_observation(state):
            if pending and pending[0] == "critic_verification_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="critic_verification_agent",
                reason="Plan policy: retrieved candidates require critic verification before recommendation.",
            )

        if state.get("critic_decision") == "ASK_USER":
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason="Plan policy: critic decision is ASK_USER; do not recommend yet.",
            )

        if (
            state.get("critic_decision") == "ACCEPT"
            and state.get("verified_candidates")
            and not state.get("last_recommendations")
        ):
            if pending and pending[0] == "recommendation_decision_agent":
                return None
            return PlannerDecision(
                action="CALL_AGENT",
                target="recommendation_decision_agent",
                reason="Plan policy: verified candidates are ready, so select recommendations before responding.",
            )

        return None

    def validate_next_action(self, state: dict, target: str) -> str | None:
        """Return a rejection reason when a planner-selected target is illegal.

        This validator is intentionally narrower than ``override``. It does
        not prescribe the next workflow step; it only protects catalogue
        grounding, basic sequencing contracts, and loop safety so the LLM
        planner remains the primary workflow controller.
        """
        if target not in ALLOWED_TARGETS:
            return f"unknown target: {target}"

        flags = set(combined_flags(state))
        missing_state_flags = {
            "underspecified_product_type",
            "missing_search_request",
            "ask_user_before_retrieval",
        }
        has_product_type = self.has_product_type(state)
        has_reflection = self.has_observation(state, "preference_reflection_agent")
        intent_label = str((state.get("intent") or {}).get("label") or "").lower()
        shopping_intents = {"recommend", "reject_or_revise", "revise", "reveal", "interpret"}

        if target == "critic_verification_agent":
            if not state.get("final_search_request") and not state.get("raw_candidates") and not state.get("previous_recommendations"):
                return "critic verification requires a search request, retrieved candidates, or existing recommendations"

        if target == "search_research_agent":
            if not has_product_type and flags.intersection(missing_state_flags):
                return "search request construction requires a recognizable product/category or clarification first"

        if target == "conserve_search_tool":
            if not state.get("final_search_request"):
                return "retrieval requires final_search_request"
            if flags.intersection(missing_state_flags) and state.get("critic_decision") != "ACCEPT":
                return "retrieval requires critic acceptance when search state is incomplete or risky"
            if state.get("critic_decision") == "ASK_USER":
                return "critic requested clarification before retrieval"

        if target == "recommendation_decision_agent":
            if not state.get("raw_candidates") and not state.get("verified_candidates"):
                return "recommendation selection requires retrieved catalogue candidates"
            if state.get("critic_decision") != "ACCEPT":
                return "recommendation selection requires critic ACCEPT after retrieval"

        if target == "response_tool":
            if (
                has_product_type
                and has_reflection
                and not state.get("final_search_request")
                and not state.get("raw_candidates")
                and intent_label in shopping_intents
            ):
                return "recognizable product request should build a search request before responding"
            if (
                state.get("raw_candidates")
                and state.get("critic_decision") == "ACCEPT"
                and state.get("verified_candidates")
                and not state.get("last_recommendations")
            ):
                return "recommendation response requires recommendation_decision_agent first"

        return None

    def validate_planner_plan(self, raw_plan: list[str], state: dict, analysis: dict | None = None) -> list[str]:
        """Light validation that preserves the planner's selected sequence."""
        cleaned: list[str] = []
        for item in raw_plan:
            target = str(item).strip()
            if target not in ALLOWED_TARGETS:
                raise ValueError(f"invalid planner target in plan: {target!r}")
            if not cleaned or cleaned[-1] != target:
                cleaned.append(target)
        if not cleaned:
            raise ValueError("LLM plan is empty")
        if len(cleaned) > 9:
            raise ValueError("LLM plan is too long")
        return cleaned

    def validate_turn_plan(self, raw_plan: list[str], state: dict, analysis: dict | None = None) -> list[str]:
        cleaned: list[str] = []
        for item in raw_plan:
            target = str(item).strip()
            if target not in ALLOWED_TARGETS:
                raise ValueError(f"invalid planner target in plan: {target!r}")
            if not cleaned or cleaned[-1] != target:
                cleaned.append(target)

        if not cleaned:
            raise ValueError("LLM plan is empty")

        analysis = analysis or {}
        if self.needs_intent_evidence(state, analysis, cleaned) and "intent_agent" not in cleaned:
            cleaned.insert(0, "intent_agent")

        if self.needs_preference_reflection(state, analysis, cleaned) and "preference_reflection_agent" not in cleaned:
            before = self._first_index(cleaned, {"search_research_agent", "conserve_search_tool", "response_tool"})
            cleaned.insert(before if before is not None else len(cleaned), "preference_reflection_agent")

        if "search_research_agent" in cleaned and "preference_reflection_agent" in cleaned:
            search_index = cleaned.index("search_research_agent")
            pref_index = cleaned.index("preference_reflection_agent")
            if pref_index > search_index:
                cleaned.pop(pref_index)
                cleaned.insert(search_index, "preference_reflection_agent")

        has_raw = state.get("raw_candidate_count", 0) > 0
        if "recommendation_decision_agent" in cleaned and "conserve_search_tool" not in cleaned and not has_raw:
            rec_index = cleaned.index("recommendation_decision_agent")
            cleaned.insert(rec_index, "conserve_search_tool")

        if "conserve_search_tool" in cleaned:
            search_index = cleaned.index("conserve_search_tool")
            critic_after_search = "critic_verification_agent" in cleaned[search_index + 1 :]
            if not critic_after_search:
                cleaned.insert(search_index + 1, "critic_verification_agent")

        if "recommendation_decision_agent" in cleaned and "critic_verification_agent" in cleaned:
            rec_index = cleaned.index("recommendation_decision_agent")
            critic_index = cleaned.index("critic_verification_agent")
            if critic_index > rec_index:
                cleaned.pop(critic_index)
                cleaned.insert(rec_index, "critic_verification_agent")

        if cleaned[-1] != "response_tool":
            cleaned.append("response_tool")

        compacted: list[str] = []
        for target in cleaned:
            if not compacted or compacted[-1] != target:
                compacted.append(target)

        if len(compacted) > 9:
            raise ValueError("LLM plan is too long")
        return compacted

    def has_observation(self, state: dict, name: str) -> bool:
        return any(obs.get("component") == name for obs in state.get("tool_observations", []))

    def needs_intent_evidence(self, state: dict, analysis: dict, plan: list[str]) -> bool:
        if state.get("intent", {}).get("label"):
            return False
        turn_type = str(analysis.get("turn_type") or "").lower()
        if turn_type in {"acceptance", "rejection", "ask_detail", "other"}:
            return True
        if "response_tool" in plan and not state.get("final_search_request") and not state.get("raw_candidates"):
            return True
        return False

    def needs_preference_reflection(self, state: dict, analysis: dict, plan: list[str]) -> bool:
        if self.has_observation(state, "preference_reflection_agent"):
            return False
        turn_type = str(analysis.get("turn_type") or "").lower()
        preference_change = str(analysis.get("preference_change") or "").lower()
        memory_turn_types = {"new_request", "clarification", "refinement", "rejection", "acceptance", "action_followup"}
        memory_changes = {"new_preference", "revision", "rejection", "acceptance", "uncertain"}
        if turn_type in memory_turn_types or preference_change in memory_changes:
            return True
        if "search_research_agent" in plan or "conserve_search_tool" in plan:
            return True
        return False

    def has_product_type(self, state: dict) -> bool:
        product_type = (state.get("preferences") or {}).get("product_type")
        if isinstance(product_type, dict):
            return bool(product_type.get("value"))
        return bool(product_type)

    def _first_index(self, values: list[str], targets: set[str]) -> int | None:
        for index, value in enumerate(values):
            if value in targets:
                return index
        return None

    def has_post_retrieval_critic_observation(self, state: dict) -> bool:
        observations = state.get("tool_observations", [])
        if not state.get("raw_candidates") and not self.has_retrieval_observation(state):
            return False
        search_index = -1
        for index, obs in enumerate(observations):
            if obs.get("component") == "conserve_search_tool":
                search_index = index
        return any(
            obs.get("component") == "critic_verification_agent"
            for obs in observations[search_index + 1 :]
        )

    def has_retrieval_observation(self, state: dict) -> bool:
        ledger = state.get("candidate_ledger") or {}
        if "raw_count" in ledger or "verified_count" in ledger:
            return True
        return any(obs.get("component") == "conserve_search_tool" for obs in state.get("tool_observations", []))


def decision_from_target(target: str, reason: str) -> PlannerDecision:
    if target in AGENTS:
        return PlannerDecision(action="CALL_AGENT", target=target, reason=reason)
    if target in TOOLS:
        return PlannerDecision(action="CALL_TOOL", target=target, reason=reason)
    raise ValueError(f"invalid planner target: {target!r}")
