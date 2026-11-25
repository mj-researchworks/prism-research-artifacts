"""Optional LLM executive planner for the bounded autonomous workflow."""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from dataclasses import asdict, is_dataclass
from typing import Any

from .flags import combined_flags
from .plan_policy import AGENTS, ALLOWED_TARGETS, TOOLS, PlanPolicy, decision_from_target
from .planner import RuleBasedPlanner
from .schemas import PlannerDecision


ALLOWED_ACTIONS = {"CALL_AGENT", "CALL_TOOL", "RESPOND", "STOP"}


class LLMExecutivePlanner:
    """Bounded planner that asks an LLM for one short plan per user turn.

    Earlier versions asked the local LLM for every micro-decision. That made
    Ollama too slow and sometimes exhausted the runner step limit. This version
    asks once per turn, validates a short target list, and then returns one
    PlannerDecision at a time from an internal queue.
    """

    def __init__(self, fallback: RuleBasedPlanner | None = None):
        self.fallback = fallback or RuleBasedPlanner()
        self.provider = os.getenv("AUTONOMOUS_PLANNER_PROVIDER", "ollama").lower()
        self.model = os.getenv("AUTONOMOUS_PLANNER_MODEL", "llama3.1:8b")
        self.endpoint = os.getenv("AUTONOMOUS_PLANNER_OLLAMA_URL", "http://localhost:11434/api/generate")
        self.deepseek_endpoint = os.getenv("DEEPSEEK_API_URL", "https://api.deepseek.com/chat/completions")
        self.deepseek_thinking = os.getenv("DEEPSEEK_THINKING", "disabled").lower()
        self.deepseek_reasoning_effort = os.getenv("DEEPSEEK_REASONING_EFFORT", "low").lower()
        self.timeout_seconds = float(os.getenv("AUTONOMOUS_PLANNER_TIMEOUT", "20"))
        self.last_backend = "not_called"
        self.last_error = ""
        self.last_usage: dict[str, Any] = {}
        self.consecutive_errors = 0
        self.disabled_after_error = False
        self.policy_mode = os.getenv("AUTONOMOUS_PLAN_POLICY_MODE", "override").strip().lower()
        self._planned_turn_id: int | None = None
        self._plan_queue: list[dict[str, str]] = []
        self._last_plan: list[str] = []
        self._last_plan_reason = ""
        self._last_turn_analysis: dict[str, Any] = {}
        self._replanned_after_rejection = False
        self.policy = PlanPolicy()

    @property
    def queued_plan(self) -> list[str]:
        return [step["target"] for step in self._plan_queue]

    @property
    def last_plan(self) -> list[str]:
        return list(self._last_plan)

    @property
    def last_plan_reason(self) -> str:
        return self._last_plan_reason

    @property
    def last_turn_analysis(self) -> dict[str, Any]:
        return dict(self._last_turn_analysis)

    def decide(self, state: dict) -> PlannerDecision:
        turn_id = state.get("turn_id")
        if self._planned_turn_id != turn_id:
            self._planned_turn_id = turn_id
            self._plan_queue = []
            self._last_plan = []
            self._last_plan_reason = ""
            self._last_turn_analysis = {}
            self._replanned_after_rejection = False

        if self.policy_mode != "validator":
            override = self.policy.override(state, _queue_targets(self._plan_queue))
            if override:
                if override.target == "response_tool":
                    self._plan_queue = []
                self.last_backend = "llm_plan_guard"
                return override

        if not self._plan_queue:
            try:
                plan, reason, analysis = self._call_model_plan(_compact_state(state))
                self._plan_queue = plan
                self._last_plan = [step["target"] for step in plan]
                self._last_plan_reason = reason
                self._last_turn_analysis = analysis
                self.last_backend = "llm_plan"
                self.last_error = ""
                self.consecutive_errors = 0
            except Exception as exc:
                self.last_backend = "rule_fallback"
                self.last_error = f"{type(exc).__name__}: {exc}"
                self.consecutive_errors += 1
                fallback_decision = self.fallback.decide(state)
                fallback_decision.reason = f"Rule fallback after LLM planner unavailable: {fallback_decision.reason}"
                return fallback_decision
        else:
            self.last_backend = "llm_plan_cached"

        if self.policy_mode != "validator":
            override = self.policy.override(state, _queue_targets(self._plan_queue))
            if override:
                if override.target == "response_tool":
                    self._plan_queue = []
                self.last_backend = "llm_plan_guard"
                return override

        step = self._plan_queue.pop(0)
        if self.policy_mode == "validator":
            rejection = self.policy.validate_next_action(state, step["target"])
            if rejection:
                return self._replan_after_rejection(state, step["target"], rejection)
        reason = step.get("why") or self._last_plan_reason or "LLM planner selected this step from the turn plan."
        return decision_from_target(step["target"], reason)

    def _replan_after_rejection(self, state: dict, target: str, rejection: str) -> PlannerDecision:
        if self._replanned_after_rejection:
            self._plan_queue = []
            self.last_backend = "llm_plan_validator_stop"
            return PlannerDecision(
                action="CALL_TOOL",
                target="response_tool",
                reason=f"Planner action remained invalid after replanning: {target}; {rejection}",
            )
        self._replanned_after_rejection = True
        compact_state = _compact_state(state)
        compact_state["planner_feedback"] = {
            "rejected_target": target,
            "reason": rejection,
            "instruction": "Choose a different legal next action. Do not repeat the rejected target until the missing state is created.",
        }
        try:
            plan, reason, analysis = self._call_model_plan(compact_state)
            self._plan_queue = plan
            self._last_plan = [step["target"] for step in plan]
            self._last_plan_reason = reason
            self._last_turn_analysis = analysis
            self.last_backend = "llm_plan_validator_replan"
            if not self._plan_queue:
                raise ValueError("replan returned an empty queue")
            step = self._plan_queue.pop(0)
            second_rejection = self.policy.validate_next_action(state, step["target"])
            if second_rejection:
                self._plan_queue = []
                self.last_backend = "llm_plan_validator_stop"
                return PlannerDecision(
                    action="CALL_TOOL",
                    target="response_tool",
                    reason=f"Planner replan selected another invalid action: {step['target']}; {second_rejection}",
                )
            return decision_from_target(
                step["target"],
                step.get("why") or f"Planner replanned after illegal action {target}: {rejection}",
            )
        except Exception as exc:
            self.last_backend = "rule_fallback"
            self.last_error = f"{type(exc).__name__}: {exc}"
            self.consecutive_errors += 1
            fallback_decision = self.fallback.decide(state)
            fallback_decision.reason = f"Rule fallback after planner validation rejection: {fallback_decision.reason}"
            return fallback_decision

    def _call_model_plan(self, compact_state: dict) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
        if self.provider == "deepseek":
            return self._call_deepseek_plan(compact_state)
        return self._call_ollama_plan(compact_state)

    def _call_ollama_plan(self, compact_state: dict) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
        prompt = _turn_plan_prompt(compact_state)
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {"temperature": 0.0, "num_predict": 700},
        }
        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.endpoint,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError("Ollama planner endpoint is unavailable") from exc

        parsed = _parse_json_object(raw.get("response", ""))
        self.last_usage = {}
        return _validate_turn_plan(parsed, compact_state, self.policy_mode)

    def _call_deepseek_plan(self, compact_state: dict) -> tuple[list[dict[str, str]], str, dict[str, Any]]:
        api_key = os.getenv("DEEPSEEK_API_KEY")
        if not api_key:
            raise RuntimeError("DEEPSEEK_API_KEY is not set")

        prompt = _turn_plan_prompt(compact_state)
        payload: dict[str, Any] = {
            "model": self.model,
            "messages": [
                {
                    "role": "system",
                    "content": "You are a bounded autonomous planner. Return only valid JSON.",
                },
                {"role": "user", "content": prompt},
            ],
            "response_format": {"type": "json_object"},
            "temperature": 0.0,
            "max_tokens": 420,
            "stream": False,
        }
        if self.deepseek_thinking in {"enabled", "disabled"}:
            thinking: dict[str, Any] = {"type": self.deepseek_thinking}
            if self.deepseek_thinking == "enabled":
                thinking["reasoning_effort"] = self.deepseek_reasoning_effort
            payload["thinking"] = thinking

        data = json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.deepseek_endpoint,
            data=data,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Authorization": f"Bearer {api_key}",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(request, timeout=self.timeout_seconds) as response:
                raw = json.loads(response.read().decode("utf-8"))
        except urllib.error.HTTPError as exc:
            body = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"DeepSeek planner HTTP {exc.code}: {body[:300]}") from exc
        except urllib.error.URLError as exc:
            raise RuntimeError("DeepSeek planner endpoint is unavailable") from exc

        try:
            content = raw["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise ValueError("DeepSeek planner response did not include message content") from exc
        self.last_usage = raw.get("usage") or {}
        parsed = _parse_json_object(content)
        return _validate_turn_plan(parsed, compact_state, self.policy_mode)



def _turn_plan_prompt(state: dict) -> str:
    return f"""You are Planner v6, the executive reasoning agent for a bounded autonomous product recommender.

Return ONLY valid JSON. Do not recommend products. Do not extract all slots yourself.
Your job is to choose the minimum specialist calls needed to transform the latest user turn into one of:
1. a clarification question,
2. a verified structured search request,
3. a retrieval/recommendation path,
4. or a final response using existing memory.

Allowed targets:
intent_agent, preference_reflection_agent, search_research_agent, conserve_search_tool, critic_verification_agent, recommendation_decision_agent, response_tool.

Autonomous control principle:
- Observe latest user turn + blackboard.
- Decide what changed, what is missing, and what risk exists.
- Call a specialist only if its output can change the blackboard or prevent an unsafe/low-quality next step.
- Prefer fewer steps when the state is already sufficient.
- Stop retrieval when the user accepts current recommendations.

Specialist roles:
- intent_agent: resolve ambiguous intent, especially accept/reject/detail/follow-up turns.
- preference_reflection_agent: update memory for new preference, clarification, revision, rejection, acceptance, or prior-option references.
- search_research_agent: build/update the structured search request and retrieval plan before retrieval.
- critic_verification_agent: check trust/readiness of a search request, retrieved candidates, or final recommendation state.
- conserve_search_tool: retrieve catalog candidates only after the search request is ready/accepted.
- recommendation_decision_agent: select final products only after retrieval and candidate verification.
- response_tool: ask clarification, acknowledge acceptance, or return the final user-facing answer.

Turn labels:
new_request = starts a shopping goal/product family.
clarification = fills missing product/category/details.
refinement = adds/changes preference, budget, use case, feature, brand, or style.
rejection = rejects brand/product/feature or asks for different options.
acceptance = accepts current recommendations, e.g. this works, looks good, I like this one.
ask_detail = asks why/details/compare about current options.
action_followup = asks to show/select/use existing options, picture, my choice, or one option using memory.
other = not a shopping action.

Decision guidance:
- Missing product/category: ask clarification; do not retrieve.
- Recognizable product request with product/category present: retrieve catalogue candidates even when preferences are sparse. Do not ask for optional preferences first.
- Availability/recommendation questions such as "do you sell X", "I want X", or "recommend X" are retrieval-oriented when X is a product type.
- Known product/category plus changed preference: update memory, update search request, verify readiness, then retrieve unless the user only asks a detail about existing recommendations.
- Rejection/revision/contradiction: reflection is needed and reasoning_depth should usually be DEEP.
- Acceptance: update memory and respond; do not retrieve again.
- Detail/follow-up using existing recommendations: usually respond from memory; retrieve only if current memory cannot answer.
- Critic is needed before retrieval/recommendation, but not for a pure clarification response.
- For recommendation-oriented turns, the normal planner-selected route is preference_reflection_agent -> search_research_agent -> critic_verification_agent -> conserve_search_tool -> critic_verification_agent -> recommendation_decision_agent -> response_tool.
- If retrieval has already happened and candidates are verified, choose recommendation_decision_agent before response.

Reasoning depth:
FAST = obvious low-risk clarification/status.
REFLECT = preference memory may change.
VERIFY = search request/candidates need trust or evidence checking.
DEEP = rejection, revision, contradiction, acceptance, weak evidence, or low confidence.

State:
{json.dumps(state, separators=(",", ":"))}

JSON shape. Keep values very short:
{{
  "turn_analysis": {{
    "turn_type": "new_request|clarification|refinement|rejection|acceptance|ask_detail|action_followup|other",
    "preference_change": "none|new_preference|revision|rejection|acceptance|uncertain",
    "retrieval_readiness": "needs_clarification|ready|not_needed",
    "search_request_status": "absent|unchanged|updated|finalized",
    "state_delta": ["short change"],
    "required_specialists": ["agent_or_tool"],
    "reasoning_depth": "FAST|REFLECT|VERIFY|DEEP",
    "quality_risk": "none|low|medium|high",
    "planner_confidence": "low|medium|high",
    "turn_interpretation": "one short sentence"
  }},
  "plan": ["intent_agent", "preference_reflection_agent", "response_tool"],
  "plan_why": {{"intent_agent": "why", "preference_reflection_agent": "why", "response_tool": "why"}},
  "reason": "short reason"
}}
"""


def _compact_state(state: dict) -> dict:
    return {
        "turn_id": state.get("turn_id"),
        "latest_user_message": state.get("latest_user_message"),
        "history_tail": state.get("history", [])[-4:],
        "intent": state.get("intent"),
        "preferences": _serialize(state.get("preferences", {})),
        "control_flags": state.get("control_flags", []),
        "preference_flags": state.get("preference_flags", []),
        "safety_flags": state.get("safety_flags", []),
        "unclassified_flags": state.get("unclassified_flags", []),
        "security_flags": combined_flags(state),
        "preferred_brands": state.get("preferred_brands", []),
        "preferred_features": state.get("preferred_features", []),
        "rejected_brands": state.get("rejected_brands", []),
        "rejected_products_count": len(state.get("rejected_products") or []),
        "last_final_search_request": _serialize(state.get("last_final_search_request")),
        "has_previous_recommendations": bool(state.get("previous_recommendations")),
        "previous_recommendation_titles": [
            str(item.get("title") or "")[:100]
            for item in (state.get("previous_recommendations") or [])[:3]
        ],
        "has_final_search_request": bool(state.get("final_search_request")),
        "final_search_request": _serialize(state.get("final_search_request")),
        "retrieval_readiness": state.get("retrieval_readiness"),
        "raw_candidate_count": len(state.get("raw_candidates") or []),
        "verified_candidate_count": len(state.get("verified_candidates") or []),
        "recommendation_count": len(state.get("last_recommendations") or []),
        "critic_decision": state.get("critic_decision"),
        "critic_reasons": state.get("critic_reasons", []),
    }


def _serialize(value: Any) -> Any:
    if is_dataclass(value):
        return asdict(value)
    if isinstance(value, dict):
        return {key: _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    return value



def _parse_json_object(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        text = text.strip("`").strip()
        if text.startswith("json"):
            text = text[4:].strip()
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("LLM did not return a JSON object")
    return json.loads(text[start : end + 1])


def _validate_turn_plan(raw: dict, state: dict, policy_mode: str = "override") -> tuple[list[dict[str, str]], str, dict[str, Any]]:
    reason = str(raw.get("reason") or "LLM planner selected a bounded turn plan.")
    analysis = raw.get("turn_analysis") or raw.get("analysis") or raw.get("semantic_analysis") or {}
    if not isinstance(analysis, dict):
        analysis = {"raw_turn_analysis": str(analysis)}
    analysis = _normalize_turn_analysis(analysis)

    plan = _extract_plan(raw)
    plan_why = raw.get("plan_why") if isinstance(raw.get("plan_why"), dict) else {}
    if plan is None:
        plan = _derive_plan_from_analysis(analysis, state)
        reason = f"Derived safe plan from Planner v3 turn analysis: {reason}"

    proposed_steps = _normalize_plan_items(plan, plan_why)
    if policy_mode == "validator":
        validated_targets = PlanPolicy().validate_planner_plan([step["target"] for step in proposed_steps], state, analysis)
    else:
        proposed_steps = _merge_required_specialists(proposed_steps, analysis, plan_why)
        validated_targets = PlanPolicy().validate_turn_plan([step["target"] for step in proposed_steps], state, analysis)
    steps_by_target = {step["target"]: step for step in proposed_steps}
    validated_steps = [
        {
            "target": target,
            "why": steps_by_target.get(target, {}).get("why") or _default_step_reason(target, analysis),
        }
        for target in validated_targets
    ]
    return validated_steps, reason, analysis


def _normalize_turn_analysis(analysis: dict) -> dict:
    normalized = dict(analysis)
    turn_aliases = {
        "refine": "refinement",
        "reject": "rejection",
        "accept": "acceptance",
        "detail": "ask_detail",
        "followup": "action_followup",
        "follow_up": "action_followup",
        "follow-up": "action_followup",
        "action": "action_followup",
    }
    turn_type = str(normalized.get("turn_type") or "other").strip().lower()
    normalized["turn_type"] = turn_aliases.get(turn_type, turn_type)

    preference_aliases = {
        "reject": "rejection",
        "accept": "acceptance",
        "refine": "new_preference",
    }
    preference_change = str(normalized.get("preference_change") or "none").strip().lower()
    normalized["preference_change"] = preference_aliases.get(preference_change, preference_change)

    readiness = str(normalized.get("retrieval_readiness") or "").strip().lower()
    if not readiness:
        if normalized["turn_type"] in {"acceptance", "ask_detail"}:
            readiness = "not_needed"
        elif normalized.get("missing_info"):
            readiness = "needs_clarification"
        elif normalized.get("needs_retrieval") is True:
            readiness = "ready"
        else:
            readiness = "needs_clarification"
    normalized["retrieval_readiness"] = readiness

    request_status = str(normalized.get("search_request_status") or "").strip().lower()
    if not request_status:
        if readiness == "needs_clarification":
            request_status = "absent"
        elif normalized["turn_type"] in {"refinement", "rejection", "clarification"}:
            request_status = "updated"
        elif readiness == "ready":
            request_status = "finalized"
        else:
            request_status = "unchanged"
    normalized["search_request_status"] = request_status

    if "state_delta" not in normalized:
        normalized["state_delta"] = []
    if "required_specialists" not in normalized:
        normalized["required_specialists"] = []
    normalized["reasoning_depth"] = _normalize_reasoning_depth(normalized)
    if "quality_risk" not in normalized:
        normalized["quality_risk"] = _infer_quality_risk(normalized)
    if "planner_confidence" not in normalized:
        normalized["planner_confidence"] = "medium"
    return normalized


def _normalize_reasoning_depth(analysis: dict) -> str:
    raw = str(analysis.get("reasoning_depth") or analysis.get("reasoning_budget") or "").strip().upper()
    aliases = {
        "LOW": "FAST",
        "LIGHT": "FAST",
        "LIGHTWEIGHT": "FAST",
        "MEDIUM": "REFLECT",
        "HIGH": "DEEP",
        "FULL": "DEEP",
        "REASON": "REFLECT",
        "CRITIC": "VERIFY",
        "VERIFY_ONLY": "VERIFY",
    }
    raw = aliases.get(raw, raw)
    if raw in {"FAST", "REFLECT", "VERIFY", "DEEP"}:
        return raw
    turn_type = str(analysis.get("turn_type") or "").lower()
    preference_change = str(analysis.get("preference_change") or "").lower()
    confidence = str(analysis.get("planner_confidence") or "").lower()
    if confidence == "low":
        return "DEEP"
    if turn_type in {"rejection", "acceptance"} or preference_change in {"rejection", "acceptance", "revision", "uncertain"}:
        return "DEEP"
    if turn_type in {"action_followup", "ask_detail"}:
        return "VERIFY"
    if turn_type in {"new_request", "refinement"} or preference_change == "new_preference":
        return "REFLECT"
    return "FAST"


def _infer_quality_risk(analysis: dict) -> str:
    depth = str(analysis.get("reasoning_depth") or "").upper()
    if depth == "DEEP":
        return "high"
    if depth in {"REFLECT", "VERIFY"}:
        return "medium"
    return "low"


def _extract_plan(raw: dict):
    for key in ("plan", "steps", "actions", "execution_plan", "workflow", "targets"):
        value = raw.get(key)
        if isinstance(value, list):
            return value
        if isinstance(value, dict):
            nested = value.get("plan") or value.get("steps") or value.get("actions") or value.get("targets")
            if isinstance(nested, list):
                return nested
    return None


def _normalize_plan_items(plan: list, plan_why: dict | None = None) -> list[dict[str, str]]:
    steps: list[dict[str, str]] = []
    aliases = {
        "intent": "intent_agent",
        "intent classifier": "intent_agent",
        "preference": "preference_reflection_agent",
        "reflection": "preference_reflection_agent",
        "search": "search_research_agent",
        "research": "search_research_agent",
        "retrieve": "conserve_search_tool",
        "retrieval": "conserve_search_tool",
        "critic": "critic_verification_agent",
        "verify": "critic_verification_agent",
        "decision": "recommendation_decision_agent",
        "recommend": "recommendation_decision_agent",
        "response": "response_tool",
        "respond": "response_tool",
    }
    plan_why = plan_why or {}
    for item in plan:
        why = "Planner selected this component for the current turn."
        if isinstance(item, dict):
            why = str(item.get("why") or item.get("purpose") or item.get("reason") or why)
            item = item.get("target") or item.get("name") or item.get("tool") or item.get("agent") or item.get("component") or item.get("action")
        target = str(item or "").strip()
        target = aliases.get(target.lower(), target)
        why = str(plan_why.get(target) or plan_why.get(target.lower()) or why)
        steps.append({"target": target, "why": why})
    return steps


def _merge_required_specialists(steps: list[dict[str, str]], analysis: dict, plan_why: dict | None = None) -> list[dict[str, str]]:
    """Treat required_specialists as the planner's own component commitments."""
    plan_why = plan_why or {}
    existing = [step["target"] for step in steps]
    merged = list(steps)
    for target in _normalize_required_specialists(analysis.get("required_specialists")):
        if target in existing:
            continue
        insert_at = _first_index(merged, {"search_research_agent", "conserve_search_tool", "critic_verification_agent", "recommendation_decision_agent", "response_tool"})
        if target == "search_research_agent":
            insert_at = _first_index(merged, {"conserve_search_tool", "critic_verification_agent", "recommendation_decision_agent", "response_tool"})
        elif target == "critic_verification_agent":
            insert_at = _first_index(merged, {"conserve_search_tool", "recommendation_decision_agent", "response_tool"})
        elif target == "response_tool":
            insert_at = len(merged)
        merged.insert(
            insert_at if insert_at is not None else len(merged),
            {
                "target": target,
                "why": str(plan_why.get(target) or _default_step_reason(target, analysis)),
            },
        )
        existing.append(target)
    return merged


def _normalize_required_specialists(value) -> list[str]:
    aliases = {
        "intent": "intent_agent",
        "intent classifier": "intent_agent",
        "preference": "preference_reflection_agent",
        "reflection": "preference_reflection_agent",
        "search": "search_research_agent",
        "research": "search_research_agent",
        "retrieve": "conserve_search_tool",
        "retrieval": "conserve_search_tool",
        "critic": "critic_verification_agent",
        "verify": "critic_verification_agent",
        "decision": "recommendation_decision_agent",
        "recommend": "recommendation_decision_agent",
        "response": "response_tool",
        "respond": "response_tool",
    }
    required = []
    items = value if isinstance(value, list) else []
    for item in items:
        target = str(item or "").strip()
        target = aliases.get(target.lower(), target)
        if target in ALLOWED_TARGETS and target not in required:
            required.append(target)
    return required


def _first_index(steps: list[dict[str, str]], targets: set[str]) -> int | None:
    for index, step in enumerate(steps):
        if step["target"] in targets:
            return index
    return None


def _derive_plan_from_analysis(analysis: dict, state: dict) -> list[dict[str, str]]:
    turn_type = str(analysis.get("turn_type") or "").lower()
    preference_change = str(analysis.get("preference_change") or "").lower()
    needs_retrieval = bool(analysis.get("needs_retrieval"))
    missing_info = analysis.get("missing_info") or []

    if turn_type in {"accept", "acceptance"} or preference_change == "acceptance":
        targets = ["intent_agent", "preference_reflection_agent", "response_tool"]
    elif missing_info and not needs_retrieval:
        targets = ["intent_agent", "preference_reflection_agent", "search_research_agent", "critic_verification_agent", "response_tool"]
    elif needs_retrieval or turn_type in {"new_request", "clarification", "refine", "refinement", "reject", "rejection", "action_followup"} or preference_change in {"new_preference", "revision", "rejection"}:
        targets = [
            "intent_agent",
            "preference_reflection_agent",
            "search_research_agent",
            "critic_verification_agent",
            "response_tool",
        ]
    elif state.get("has_previous_recommendations"):
        targets = ["intent_agent", "preference_reflection_agent", "response_tool"]
    else:
        targets = ["intent_agent", "preference_reflection_agent", "search_research_agent", "response_tool"]
    return [{"target": target, "why": _default_step_reason(target, analysis)} for target in targets]


def _default_step_reason(target: str, analysis: dict) -> str:
    interpretation = analysis.get("turn_interpretation") or analysis.get("state_change_expected") or "the current turn"
    reasons = {
        "intent_agent": f"Classify intent for {interpretation}.",
        "preference_reflection_agent": f"Update preference memory for {interpretation}.",
        "search_research_agent": "Build or update the structured search request from blackboard state.",
        "conserve_search_tool": "Retrieve grounded catalog candidates because the planner says retrieval is needed.",
        "critic_verification_agent": "Validate state, evidence, and candidates before response or recommendation.",
        "recommendation_decision_agent": "Select ranked recommendations from verified candidates.",
        "response_tool": "Return the final user-facing response for this turn.",
    }
    return reasons.get(target, "Planner selected this component for the current turn.")


def _queue_targets(queue: list[dict[str, str]]) -> list[str]:
    return [step["target"] for step in queue]


# Kept for compatibility with older imports/tests that expected one decision.
def _planner_prompt(state: dict) -> str:
    return _turn_plan_prompt(state)


def _validate_decision(raw: dict) -> PlannerDecision:
    action = raw.get("action")
    target = raw.get("target")
    reason = str(raw.get("reason") or "LLM planner selected next action.")

    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"invalid planner action: {action!r}")
    if action in {"CALL_AGENT", "CALL_TOOL"} and target not in ALLOWED_TARGETS:
        raise ValueError(f"invalid planner target: {target!r}")
    if action in {"RESPOND", "STOP"} and target not in {None, "response_tool"}:
        raise ValueError(f"invalid planner target for {action}: {target!r}")

    return PlannerDecision(action=action, target=target, reason=reason)
