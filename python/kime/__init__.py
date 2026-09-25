"""kime: fast calibrated decisions from an encoder.

This package runs the same engine as `kime serve` inside the Python process. `kime.load` and
`kime.Agent` take the arguments Laya's do and answer in Laya's shape, so code written for
`laya.load(...)` works after changing the import.

    import kime
    agent = kime.load("convaiinnovations/laya")
    agent.system_one("I was charged twice", kime.triage_questions())
"""

import json
import os
from typing import Any, Dict, List, Optional, Union

from . import _native

__version__ = _native.__version__

__all__ = [
    "Agent",
    "RLAgent",
    "AgentStep",
    "load",
    "agent_step",
    "clean_email_body",
    "email_questions",
    "email_state",
    "guard_questions",
    "moderation_questions",
    "router_questions",
    "triage_questions",
    "Router",
    "RouteDecision",
    "DEFAULT_MODELS",
    "detect_language",
    "detect_script",
    "is_english",
]

_LAYA_REPO = "convaiinnovations/laya"

State = Union[str, dict, list]


def _model_name(model_id_or_path: str, subfolder: Optional[str]) -> str:
    """The kime name for a Laya model id: a local path, an alias or an hf:// name."""
    if os.path.exists(model_id_or_path):
        return os.path.join(model_id_or_path, subfolder) if subfolder else model_id_or_path
    name = model_id_or_path
    if name.startswith("hf://"):
        name = name[len("hf://"):]
    if name == _LAYA_REPO:
        return "laya-" + subfolder if subfolder else "laya"
    if "/" not in name:
        # Already a kime name, such as "laya" or "laya-multilingual".
        return name
    return "hf://" + name + ("/" + subfolder if subfolder else "")


def _unsupported(**kwargs: Any) -> None:
    for k, v in kwargs.items():
        if v is not None and v is not False:
            raise NotImplementedError("kime does not support %s= yet" % k)


class Agent:
    """A loaded model with Laya's `Agent` API.

    `device` is "auto", "cpu", "cuda" or "cuda:N". `threads` sets the CPU threads, all cores when
    0. `precision` is "f16", "f32" or "int8". Laya's `fast` and `compile` are accepted and do
    nothing, since kime always runs its fast path. Hooks and `lang_temperatures` are not supported
    yet and raise NotImplementedError.
    """

    def __init__(
        self,
        model_id_or_path: str = _LAYA_REPO,
        device: Optional[str] = None,
        token: Optional[str] = None,
        subfolder: Optional[str] = None,
        fast: bool = False,
        compile: bool = False,
        lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: bool = True,
        hooks_concurrent: bool = True,
        *,
        threads: int = 0,
        precision: Optional[str] = None,
    ):
        _unsupported(
            lang_temperatures=lang_temperatures,
            hooks=hooks,
            on_predict_start=on_predict_start,
            on_predict_end=on_predict_end,
        )
        # kime never downloads, so the token is not needed: `kime pull` fetches models.
        del token, fast, compile, hooks_raise, hooks_concurrent
        self.model_id = model_id_or_path
        self._engine = _native.Engine(
            _model_name(model_id_or_path, subfolder),
            device=device or "auto",
            threads=threads,
            precision=precision,
        )

    @property
    def device(self) -> str:
        return self._engine.device

    def system_one(
        self,
        state: State,
        questions: Dict[str, Dict[str, Any]],
        lang: Optional[str] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
    ) -> Dict[str, Any]:
        """Answers every question about `state` in one forward pass."""
        _unsupported(
            lang=lang,
            hooks=hooks,
            on_predict_start=on_predict_start,
            on_predict_end=on_predict_end,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        body = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
        return json.loads(self._engine.system_one(body))

    predict = system_one

    def predict_batch(
        self,
        states: List[State],
        questions: Dict[str, Dict[str, Any]],
        batch_size: Optional[int] = None,
        lang: Optional[str] = None,
        hooks=None,
        on_predict_start=None,
        on_predict_end=None,
        hooks_raise: Optional[bool] = None,
        max_len: Optional[int] = None,
        head_max_len: Optional[int] = None,
        sort_by_length: bool = False,
    ) -> List[Dict[str, Any]]:
        """Answers the same questions about many states. kime packs the forward passes itself, so
        `batch_size` only bounds how many states go to the engine at once and `sort_by_length`
        changes nothing."""
        _unsupported(
            lang=lang,
            hooks=hooks,
            on_predict_start=on_predict_start,
            on_predict_end=on_predict_end,
            max_len=max_len,
            head_max_len=head_max_len,
        )
        q = json.dumps(questions, ensure_ascii=False)
        bodies = [
            '{"state":%s,"questions":%s}' % (json.dumps(s, ensure_ascii=False), q) for s in states
        ]
        step = batch_size or len(bodies) or 1
        out: List[Dict[str, Any]] = []
        for i in range(0, len(bodies), step):
            out.extend(json.loads(r) for r in self._engine.predict_batch(bodies[i : i + step]))
        return out

    def count_tokens(self, state: State, questions: Dict[str, Dict[str, Any]]) -> int:
        """The tokens the model reads for these questions about `state`."""
        body = json.dumps({"state": state, "questions": questions}, ensure_ascii=False)
        return self._engine.count_tokens(body)

    def __enter__(self) -> "Agent":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        return False

    def __repr__(self) -> str:
        return "kime.Agent(%r, device=%r)" % (self.model_id, self.device)


RLAgent = Agent


def load(
    model_id_or_path: str = _LAYA_REPO,
    device: Optional[str] = None,
    token: Optional[str] = None,
    subfolder: Optional[str] = None,
    fast: bool = False,
    lang_temperatures: Optional[Dict[str, Dict[str, Any]]] = None,
    hooks=None,
    on_predict_start=None,
    on_predict_end=None,
    hooks_raise: bool = True,
    hooks_concurrent: bool = True,
    *,
    threads: int = 0,
    precision: Optional[str] = None,
) -> Agent:
    """Loads a model, as `laya.load` does. The model has to be on disk: see `kime pull`."""
    return Agent(
        model_id_or_path,
        device=device,
        token=token,
        subfolder=subfolder,
        fast=fast,
        lang_temperatures=lang_temperatures,
        hooks=hooks,
        on_predict_start=on_predict_start,
        on_predict_end=on_predict_end,
        hooks_raise=hooks_raise,
        hooks_concurrent=hooks_concurrent,
        threads=threads,
        precision=precision,
    )


def clean_email_body(body: str, max_chars: int = 3000) -> str:
    """Removes quoted history, signatures and disclaimers, as Laya's does."""
    return _native.clean_email_body(body or "", max_chars)


def email_state(
    subject: str, body: str, sender: Optional[str] = None, clean: bool = True, **extra: Any
) -> Dict[str, Any]:
    """A state for email questions, as Laya builds it."""
    state = {
        "subject": (subject or "").strip(),
        "body": clean_email_body(body) if clean else (body or ""),
    }
    if sender:
        state["from"] = sender
    state.update({k: v for k, v in extra.items() if v is not None})
    return state


def triage_questions() -> Dict[str, Any]:
    return json.loads(_native.preset("triage"))


def email_questions(categories: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    c = None if categories is None else json.dumps(categories, ensure_ascii=False)
    return json.loads(_native.preset("email", c))


def guard_questions() -> Dict[str, Any]:
    return json.loads(_native.preset("guard"))


def moderation_questions() -> Dict[str, Any]:
    return json.loads(_native.preset("moderation"))


def router_questions() -> Dict[str, Any]:
    return json.loads(_native.preset("router"))


class AgentStep:
    """The questions for one browser agent step, built as jev-ultrafast builds them, and the
    reading of the answers into an action."""

    def __init__(self, snapshot: Dict[str, Any], goal: str, history: Optional[List[Any]] = None):
        self._step = _native.AgentStep(
            json.dumps(snapshot, ensure_ascii=False),
            goal,
            json.dumps(history or [], ensure_ascii=False),
        )
        self.state = json.loads(self._step.state)
        self.questions = json.loads(self._step.questions)

    def body(self, model: str = "jev-latest") -> Dict[str, Any]:
        """The request body for `POST /v1/systemone`."""
        return json.loads(self._step.body(model))

    def decide(self, answers: Dict[str, Any]) -> Dict[str, Any]:
        """The action to run: `choice`, `operation`, `target`, `confidence`, `probabilities` and
        `target_confidence`. Raises ValueError for an answer jev-ultrafast would not act on."""
        return json.loads(self._step.decide(json.dumps(answers)))

    def run(self, agent: Agent) -> Dict[str, Any]:
        """Asks `agent` and reads its answers."""
        return self.decide(agent.system_one(self.state, self.questions)["answers"])


def agent_step(
    snapshot: Dict[str, Any], goal: str, history: Optional[List[Any]] = None
) -> AgentStep:
    return AgentStep(snapshot, goal, history)


from .router import DEFAULT_MODELS, RouteDecision, Router, detect_language, detect_script, is_english  # noqa: E402
