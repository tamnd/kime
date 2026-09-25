"""Laya's `Router`: it picks the English, multilingual or typed-decisions checkpoint for each
request and loads them as they are needed.

The arguments, the precedence and the `RouteDecision` are Laya's. The one difference is the
detection step: by default it is the one `kime serve` uses, Laya's word lists with a language
identifier on top, so a German sentence with no accents or an English one about a crème brûlée
goes where it should. `Router(identifier=False)` gives Laya's word lists alone.
"""

import json
import threading
from collections.abc import Sequence as SequenceABC
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from . import _native

BUNDLE_REPO = "convaiinnovations/laya"
DEFAULT_MODELS = {
    "english": (BUNDLE_REPO, None),
    "multilingual": (BUNDLE_REPO, "multilingual"),
    "typed-decisions": (BUNDLE_REPO, "typed-decisions"),
}
STANDALONE_MODELS = {
    "english": "convaiinnovations/laya",
    "multilingual": "convaiinnovations/laya-multilingual",
    "typed-decisions": "convaiinnovations/laya-typed-decisions",
}

_ALIASES = {
    "en": "english", "laya": "english", "default": "english",
    "multi": "multilingual", "ml": "multilingual", "laya-multilingual": "multilingual",
    "typed": "typed-decisions", "typed_decisions": "typed-decisions",
    "laya-typed-decisions": "typed-decisions", "decisions": "typed-decisions",
}

# The question ids of the four workflows the typed-decisions checkpoint was tuned on.
_TYPED_DECISION_WORKFLOWS = {
    "agent_trace_observability": {"action", "needs_review", "outcome", "risk", "urgency"},
    "customer_service": {"action", "category", "churn_risk", "needs_human", "urgency"},
    "invoice_processing": {"discrepancy_severity", "disposition", "duplicate", "matches_order", "urgency"},
    "security_incidents": {"credential_compromise", "disposition", "severity", "true_positive", "urgency"},
}

_ENGLISH_SUBTAGS = ("en", "eng", "english")

State = Union[str, dict, list, None]


def _split(spec: Any) -> Tuple[str, Optional[str]]:
    if isinstance(spec, (tuple, list)):
        repo, sub = (list(spec) + [None])[:2]
        return repo, sub
    return spec, None


def _repo_str(spec: Any) -> str:
    repo, sub = _split(spec)
    return "%s/%s" % (repo, sub) if sub else repo


class RouteDecision(dict):
    """Which checkpoint, why, and what detection found. A dict, so it goes straight into JSON."""

    @property
    def model(self) -> str:
        return self["model"]

    @property
    def reason(self) -> str:
        return self["reason"]

    def __repr__(self) -> str:
        return "RouteDecision(model=%r, reason=%r)" % (self["model"], self["reason"])


def normalise_name(name: str) -> str:
    key = str(name).strip().lower()
    key = _ALIASES.get(key, key)
    if key not in DEFAULT_MODELS:
        raise ValueError(
            "unknown model %r; choose one of %s (or an alias: %s)"
            % (name, sorted(DEFAULT_MODELS), sorted(_ALIASES))
        )
    return key


def match_typed_decisions_workflow(questions: Dict[str, Any]) -> Optional[str]:
    """The typed-decisions workflow whose question ids these are exactly, else None."""
    ids = set(questions or {})
    for wf, sig in _TYPED_DECISION_WORKFLOWS.items():
        if ids == sig:
            return wf
    return None


def _english_from_code(value: Any) -> Optional[bool]:
    if value is None:
        return None
    code = str(value).strip().lower()
    if not code:
        return None
    code = code.split(".", 1)[0]
    primary = code.replace("_", "-").split("-", 1)[0]
    if not primary:
        return None
    return primary in _ENGLISH_SUBTAGS


def detect_language(state: State) -> Dict[str, Any]:
    """Laya's `analyse`: the script, the language when the word lists can tell, and whether the
    English checkpoint can read the state."""
    return json.loads(_native.analyse(json.dumps(state, ensure_ascii=False)))


def detect_script(text: str) -> str:
    """The script most of the letters are in, or "unknown" when there are none."""
    return _native.detect_script(text)


def guess_latin_language(text: str) -> Optional[str]:
    """Laya's language code for Latin script text from its word lists, or None when undecided."""
    return _native.guess_latin_language(text)


def state_text(state: State, max_chars: int = 4000) -> str:
    """The string leaves of a state joined by spaces, at most `max_chars` characters, which is
    the text detection reads. Keys are left out."""
    return _native.state_text(json.dumps(state, ensure_ascii=False), max_chars)


def is_english(state: State) -> bool:
    """Whether Laya's word lists say the English checkpoint can read the state."""
    return bool(detect_language(state)["is_english"])


class Router:
    """Loads checkpoints as they are needed and sends each request to the right one.

        r = kime.Router()
        r.predict({"message": "Mein Konto wurde zweimal belastet"}, questions)   # multilingual
        r.predict({"message": "I was charged twice"}, questions)                 # english

    `max_loaded` caps how many checkpoints stay loaded, dropping the least recently used.
    `threads` and `precision` go to every `kime.Agent` the router builds.
    """

    def __init__(
        self,
        models: Optional[Dict[str, Any]] = None,
        device: Optional[str] = None,
        token: Optional[str] = None,
        max_loaded: int = 2,
        default: str = "english",
        auto_task_detection: bool = False,
        standalone_repos: bool = False,
        preload: bool = False,
        lang_guess: Optional[Any] = None,
        *,
        identifier: bool = True,
        threads: int = 0,
        precision: Optional[str] = None,
    ):
        self.models = dict(STANDALONE_MODELS if standalone_repos else DEFAULT_MODELS)
        if models:
            self.models.update({normalise_name(k): v for k, v in models.items()})
        self.device = device
        self.token = token
        self.max_loaded = max(1, int(max_loaded))
        self.default = normalise_name(default)
        self.auto_task_detection = bool(auto_task_detection)
        self.lang_guess = lang_guess
        self.identifier = bool(identifier)
        self.threads = threads
        self.precision = precision
        self._agents: Dict[str, Any] = {}
        self._order: List[str] = []
        self._lock = threading.RLock()
        if preload:
            self.preload()

    def load(self, name: str):
        """The `kime.Agent` for `name`, built the first time it is asked for."""
        from . import Agent

        key = normalise_name(name)
        with self._lock:
            if key in self._agents:
                self._touch(key)
                return self._agents[key]
            repo, sub = _split(self.models[key])
            # threads and precision go only when set, so a Laya style Agent stand in still builds.
            extra = {k: v for k, v in (("threads", self.threads), ("precision", self.precision)) if v}
            agent = Agent(repo, device=self.device, token=self.token, subfolder=sub, **extra)
            self._agents[key] = agent
            self._order.append(key)
            self._evict()
            return agent

    def _touch(self, key: str) -> None:
        with self._lock:
            if key in self._order:
                self._order.remove(key)
            self._order.append(key)

    def _evict(self) -> None:
        with self._lock:
            while len(self._order) > self.max_loaded:
                self._agents.pop(self._order.pop(0), None)
            for k in list(self._agents):
                if k not in self._order:
                    self._agents.pop(k, None)

    def attach(self, name: str, agent: Any):
        """Registers an agent already loaded, instead of loading a second copy."""
        key = normalise_name(name)
        with self._lock:
            self._agents[key] = agent
            self._touch(key)
            self.max_loaded = max(self.max_loaded, len(self._agents))
        return agent

    def preload(self, names: Optional[List[str]] = None) -> "Router":
        """Loads checkpoints now, all of them by default, so no request waits for one."""
        names = [normalise_name(n) for n in (list(self.models) if names is None else names)]
        with self._lock:
            self.max_loaded = max(self.max_loaded, len(set(names) | set(self._agents)))
            for n in names:
                if n not in self._agents:
                    self.load(n)
        return self

    def unload(self, name: Optional[str] = None) -> None:
        """Drops one checkpoint, or all of them."""
        with self._lock:
            if name is None:
                self._agents.clear()
                self._order.clear()
            else:
                key = normalise_name(name)
                self._agents.pop(key, None)
                if key in self._order:
                    self._order.remove(key)

    @property
    def loaded(self) -> List[str]:
        with self._lock:
            return list(self._order)

    def _resolve_hint(self, hint: Any, state: State) -> Optional[bool]:
        if hint is None:
            return None
        if callable(hint):
            hint = hint(state)
        return _english_from_code(hint)

    def _decision(self, key: str, reason: str, detection: Any, workflow: Optional[str]) -> RouteDecision:
        return RouteDecision(
            model=key,
            repo=_repo_str(self.models[key]),
            reason=reason,
            detection=detection,
            workflow=workflow,
        )

    def route(
        self,
        state: State,
        questions: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
    ) -> RouteDecision:
        """Picks the checkpoint without loading or running anything. The order is Laya's: `model`,
        then `task`, then a typed-decisions workflow if `auto_task_detection` is on, then `lang`,
        then `lang_guess`, then detection, then the default."""
        if model is not None:
            key = normalise_name(model)
            return self._decision(key, "explicit model=%r" % model, None, None)
        if task is not None:
            typed = str(task).lower().replace("-", "_") == "typed_decisions"
            key = normalise_name("typed-decisions" if typed else task)
            return self._decision(key, "explicit task=%r" % task, None, None)
        workflow = match_typed_decisions_workflow(questions or {})
        if workflow and self.auto_task_detection:
            reason = "question ids match the %r typed-decisions workflow" % workflow
            return self._decision("typed-decisions", reason, None, workflow)
        # A blank code names no language, so it falls through to the hints and detection.
        english = None if lang is None else _english_from_code(lang)
        if english is not None:
            key = "english" if english else "multilingual"
            return self._decision(key, "explicit lang=%r" % lang, None, workflow)
        for source, hint in (("lang_guess", lang_guess), ("Router(lang_guess=...)", self.lang_guess)):
            resolved = self._resolve_hint(hint, state)
            if resolved is not None:
                key = "english" if resolved else "multilingual"
                what = "English" if resolved else "non-English"
                reason = "%s: the caller identified this as %s text" % (source, what)
                return self._decision(key, reason, None, workflow)
        if self.identifier:
            key, reason, det = self._detect(state)
        else:
            key, reason, det = self._word_lists(state)
        return self._decision(key, reason, det, workflow)

    def _detect(self, state: State) -> Tuple[str, str, Any]:
        d = json.loads(_native.detect(json.dumps(state, ensure_ascii=False), self.default == "english"))
        if d["english"] is None:
            # The native side names the default as english or multilingual, and it may be
            # typed-decisions here.
            head = d["reason"].rsplit("using default (", 1)[0]
            return self.default, "%susing default (%s)" % (head, self.default), d["detection"]
        return ("english" if d["english"] else "multilingual"), d["reason"], d["detection"]

    def _word_lists(self, state: State) -> Tuple[str, str, Any]:
        det = detect_language(state)
        if det["script"] == "unknown":
            key = self.default
            reason = "no letters detected in state; using default (%s)" % key
        elif det["script"] != "latin":
            key = "multilingual"
            reason = "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it" % (
                det["script"], 100 * float(det["non_latin_fraction"]))
        elif not det["is_english"]:
            key = "multilingual"
            if det["language"]:
                reason = "Latin script but language looks like %r, not English" % det["language"]
            else:
                reason = (
                    "Latin script, language not identified but %.0f%% non-English letters; "
                    "not safe for the English checkpoint" % (100 * float(det["diacritic_rate"]))
                )
        elif det["language_undecided"]:
            key = self.default
            reason = (
                "Latin script, language not identified and no non-English letters; "
                "using default (%s)" % key
            )
        else:
            key = "english"
            reason = "English Latin text"
        return key, reason, det

    def predict(
        self,
        state: State,
        questions: Dict[str, Any],
        model: Optional[str] = None,
        task: Optional[str] = None,
        lang: Optional[str] = None,
        lang_guess: Optional[Any] = None,
    ) -> Dict[str, Any]:
        """Routes, then answers every question on the chosen checkpoint. The result has Laya's
        `routing` key with the decision."""
        decision = self.route(state, questions, model=model, task=task, lang=lang, lang_guess=lang_guess)
        result = self.load(decision["model"]).system_one(state, questions)
        result["routing"] = dict(decision)
        return result

    system_one = predict

    def route_batch(self, requests: Sequence[Dict[str, Any]]) -> List[RouteDecision]:
        """Routes every request without loading a checkpoint, in input order. Each request is a
        dict with `state` and `questions` and may have `model`, `task`, `lang` and `lang_guess`,
        as `route` takes them."""
        if not isinstance(requests, SequenceABC) or isinstance(requests, (str, bytes)):
            raise TypeError("requests must be a sequence of request dictionaries")
        decisions = []
        for i, request in enumerate(requests):
            if not isinstance(request, dict):
                raise TypeError("request %d must be a dict, got %s" % (i, type(request).__name__))
            for key in ("state", "questions"):
                if key not in request:
                    raise ValueError("request %d is missing required key %r" % (i, key))
            questions = request["questions"]
            if not isinstance(questions, dict):
                raise TypeError("request %d 'questions' must be a dict, got %s" % (i, type(questions).__name__))
            decisions.append(self.route(request["state"], questions, model=request.get("model"),
                                        task=request.get("task"), lang=request.get("lang"),
                                        lang_guess=request.get("lang_guess")))
        return decisions

    def predict_batch(self, requests: Sequence[Dict[str, Any]], batch_size: Optional[int] = None) -> List[Dict[str, Any]]:
        """Routes every request, then answers them with one `predict_batch` per checkpoint and
        question set, so each checkpoint loads at most once. Results come back in input order,
        each with its `routing`. Question sets that differ only in key order are kept apart,
        since options are positional."""
        decisions = self.route_batch(requests)
        groups: Dict[str, List[int]] = {}
        for i, decision in enumerate(decisions):
            groups.setdefault(decision["model"], []).append(i)
        results: List[Any] = [None] * len(decisions)
        for name, indices in groups.items():
            agent = self.load(name)
            by_questions: Dict[str, List[int]] = {}
            for i in indices:
                key = json.dumps(requests[i]["questions"], ensure_ascii=False, default=str)
                by_questions.setdefault(key, []).append(i)
            for members in by_questions.values():
                states = [requests[i]["state"] for i in members]
                out = agent.predict_batch(states, requests[members[0]]["questions"], batch_size=batch_size)
                if len(out) != len(members):
                    raise RuntimeError("internal error: Agent.predict_batch returned %d results for %d states"
                                       % (len(out), len(members)))
                for i, result in zip(members, out):
                    result["routing"] = dict(decisions[i])
                    results[i] = result
        return results

    predict_many = predict_batch

    def decide(self, state: State, schema: Any = None, *, questions: Optional[Dict[str, Any]] = None,
               return_details: bool = False, **predict_kwargs: Any) -> Any:
        """Routes, then answers against a schema, as Laya's `Router.decide` does. The details
        carry the routing decision."""
        from .structured import decide as _decide

        return _decide(self, state, schema, questions=questions, return_details=return_details, **predict_kwargs)

    def __enter__(self) -> "Router":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        self.unload()
        return False

    def __repr__(self) -> str:
        return "Router(loaded=%s, max_loaded=%d, default=%r)" % (self.loaded, self.max_loaded, self.default)
