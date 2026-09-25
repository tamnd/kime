"""The TypeSafe style client: the names, arguments, answers, errors and retry rules of
`typesafe_sdk` 0.7.1, so code written for it runs against kime by changing the import.

    from kime import TypeSafeClient, Choice, Noul

    client = TypeSafeClient(base_url="http://127.0.0.1:8000")    # kime serve, Jev or impossibl
    client = TypeSafeClient(local="laya")                         # in process, no HTTP
    r = client.system_one("I was charged twice", {"billing": Noul(instructions="Is this about billing?")})
    r.nouls["billing"].noul, r.request_id, r.usage

It needs nothing outside the standard library. The differences from `typesafe_sdk`:

- `local=` answers in process with a model name or path, or a `kime.Agent`, the way `kime serve`
  answers a Jev client, with no HTTP and no key.
- The key is only required for api.typesafe.ai, since `kime serve` runs without keys by default.
- The base URL is `http://127.0.0.1:8000` when neither `base_url` nor an environment variable
  names one.
- `KIME_API_KEY`, `KIME_BASE_URL`, `KIME_DEFAULT_MODEL` and `KIME_LOG_LEVEL` come first, and the
  `TYPESAFE_` ones are read when they are not set.
- `http_client` is not supported, since there is no httpx. `transport` takes an object with a
  `handle(method, url, headers, content, timeout)` method that returns `(status, headers, body)`.
- Questions and answers are plain classes rather than Pydantic models. `response_model` still
  takes a Pydantic model when Pydantic is installed.
"""

import asyncio
import http.client
import json
import logging
import math
import os
import platform
import random
import socket
import sys
import threading
import time
import uuid
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Sequence, Set, Tuple, Type, Union
from urllib.parse import urlsplit

API_KEY_ENV = "KIME_API_KEY"
BASE_URL_ENV = "KIME_BASE_URL"
DEFAULT_MODEL_ENV = "KIME_DEFAULT_MODEL"
LOG_LEVEL_ENV = "KIME_LOG_LEVEL"
TYPESAFE_ENV = {
    API_KEY_ENV: "TYPESAFE_API_KEY",
    BASE_URL_ENV: "TYPESAFE_BASE_URL",
    DEFAULT_MODEL_ENV: "TYPESAFE_DEFAULT_MODEL",
    LOG_LEVEL_ENV: "TYPESAFE_LOG_LEVEL",
}
TYPESAFE_BASE_URL = "https://api.typesafe.ai"
DEFAULT_BASE_URL = "http://127.0.0.1:8000"
DEFAULT_MODEL = "jev-latest"
DEFAULT_TIMEOUT = 10.0

SYSTEM_ONE_PATH = "/v1/systemone"
MODELS_PATH = "/v1/models"
MAX_ERROR_BODY_LENGTH = 200
REQUEST_ID_HEADER = "x-typesafe-request-id"
RETRY_COUNT_HEADER = "X-TypeSafe-Retry-Count"
SECRET_HEADERS = frozenset({"authorization", "proxy-authorization", "x-api-key", "api-key", "cookie", "set-cookie"})

JSONValue = Any
JSONContent = Union[str, Mapping[str, Any], Sequence[Any]]

logger = logging.getLogger("kime.typesafe")


def _version() -> str:
    try:
        from importlib.metadata import version

        return version("kime")
    except Exception:
        return "0"


SDK = "kime-python/%s" % _version()
RUNTIME = "python/%s (%s; %s)" % (platform.python_version(), sys.platform, platform.machine())


def _env(name: str) -> str:
    return os.environ.get(name, "").strip() or os.environ.get(TYPESAFE_ENV[name], "").strip()


_level = _env(LOG_LEVEL_ENV)
if _level and not logger.handlers:
    logger.addHandler(logging.StreamHandler())
    logger.setLevel(_level.upper())


# Headers


class Headers(Mapping[str, str]):
    """Response or request headers, looked up without regard to case."""

    def __init__(self, items: Union[Mapping[str, str], Sequence[Tuple[str, str]], None] = None):
        self._d: Dict[str, Tuple[str, str]] = {}
        if items is not None:
            self.update(items)

    def update(self, items: Union[Mapping[str, str], Sequence[Tuple[str, str]]]) -> None:
        pairs = items.items() if isinstance(items, Mapping) else items
        for k, v in pairs:
            self[k] = v

    def __setitem__(self, key: str, value: str) -> None:
        self._d[key.lower()] = (key, str(value))

    def __getitem__(self, key: str) -> str:
        return self._d[key.lower()][1]

    def pop(self, key: str, default: Any = None) -> Any:
        v = self._d.pop(key.lower(), None)
        return default if v is None else v[1]

    def __iter__(self) -> Iterator[str]:
        return (k for k, _ in self._d.values())

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        shown = {k: ("[secure]" if k.lower() in SECRET_HEADERS else v) for k, v in self._d.values()}
        return "Headers(%r)" % shown


# Errors


def parse_retry_after(headers: Mapping[str, str]) -> Optional[float]:
    """The wait the server asked for in milliseconds, from `retry-after-ms` or `retry-after`."""
    for name, multiplier in (("retry-after-ms", 1), ("retry-after", 1000)):
        raw = headers.get(name)
        if raw is None:
            continue
        try:
            value = float(raw.strip() or "0")
        except ValueError:
            if name == "retry-after":
                try:
                    return max(0.0, (parsedate_to_datetime(raw).timestamp() - time.time()) * 1000)
                except (ValueError, TypeError, OverflowError):
                    pass
        else:
            if math.isfinite(value):
                if value >= 0:
                    delay = value * multiplier
                    if math.isfinite(delay):
                        return delay
                elif name == "retry-after":
                    return None
    return None


def extract_message(body: Any) -> Optional[str]:
    if isinstance(body, str):
        return body or None
    if not isinstance(body, dict):
        return None
    error, message, detail = body.get("error"), body.get("message"), body.get("detail")
    if isinstance(error, str):
        return error
    if isinstance(error, dict) and isinstance(error.get("message"), str):
        return error["message"]
    if isinstance(message, str):
        return message
    if isinstance(detail, str):
        return detail
    if isinstance(detail, dict) and isinstance(detail.get("message"), str):
        return detail["message"]
    if isinstance(detail, list):
        parts = []
        for entry in detail:
            if not isinstance(entry, dict) or not isinstance(entry.get("msg"), str):
                continue
            loc = entry.get("loc")
            path = ".".join(str(i) for i in loc if i != "body") if isinstance(loc, list) else ""
            parts.append("%s: %s" % (path, entry["msg"]) if path else entry["msg"])
        return "; ".join(parts) or None
    return None


def _dumps(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), default=_encode).encode()


class TypeSafeError(Exception):
    """Base exception for SDK failures."""


class TypeSafeAPIError(TypeSafeError):
    """An unsuccessful HTTP response with its body and request metadata."""

    def __init__(self, status: int, body: Any, headers: Mapping[str, str], message: Optional[str] = None,
                 endpoint: Optional[str] = None) -> None:
        super().__init__(status, body, headers, message, endpoint)
        self.status = status
        self.body = body
        self.headers = headers
        self.endpoint = endpoint
        if message is None:
            detail = extract_message(body)
            if detail:
                message = detail
            elif body is None:
                message = "status code (no body)"
            else:
                raw = body if isinstance(body, str) else _dumps(body).decode()
                message = raw[:MAX_ERROR_BODY_LENGTH] + "…" if len(raw) > MAX_ERROR_BODY_LENGTH else raw
        self._message = message

    def __str__(self) -> str:
        message = "%s %s" % (self.status, self._message) if self._message else str(self.status)
        if self.endpoint is not None:
            message = "%s: %s" % (self.endpoint, message)
        if self.request_id is not None:
            message += " (request_id=%s)" % self.request_id
        return message

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, str(self))

    @property
    def request_id(self) -> Optional[str]:
        return self.headers.get(REQUEST_ID_HEADER)


class TypeSafeBadRequestError(TypeSafeAPIError):
    """The request was invalid (400)."""


class TypeSafeAuthenticationError(TypeSafeAPIError):
    """Authentication failed (401)."""


class TypeSafePermissionDeniedError(TypeSafeAPIError):
    """Access was denied (403)."""


class TypeSafeNotFoundError(TypeSafeAPIError):
    """The resource was not found (404)."""


class TypeSafeUnprocessableEntityError(TypeSafeAPIError):
    """The request failed server validation (422)."""


class TypeSafeRateLimitError(TypeSafeAPIError):
    """The rate limit was exceeded (429)."""

    def __init__(self, status: int, body: Any, headers: Mapping[str, str], message: Optional[str] = None,
                 endpoint: Optional[str] = None) -> None:
        super().__init__(status, body, headers, message, endpoint)
        self.retry_after_ms = parse_retry_after(headers)


class TypeSafeInternalServerError(TypeSafeAPIError):
    """The server failed to process the request (5xx)."""


class TypeSafeAPIConnectionError(TypeSafeError, ConnectionError):
    """A request failed without an HTTP response."""


class TypeSafeAPITimeoutError(TypeSafeAPIConnectionError, TimeoutError):
    """A request exceeded its configured timeout."""

    def __init__(self, timeout: float) -> None:
        super().__init__(timeout)
        self.timeout = timeout

    def __str__(self) -> str:
        return "Request timed out (timeout=%s)." % (self.timeout,)

    def __repr__(self) -> str:
        return "%s(%r)" % (type(self).__name__, str(self))


class TypeSafeAPIResponseValidationError(TypeSafeAPIError):
    """A successful HTTP response whose body was missing or structurally invalid required data."""

    def __init__(self, status: int, body: Any, headers: Mapping[str, str], field_path: str,
                 endpoint: Optional[str] = None) -> None:
        self.field_path = field_path
        super().__init__(status, body, headers, "Invalid response data at %r." % field_path, endpoint)
        self.args = (status, body, headers, field_path, endpoint)


STATUS_ERROR_TYPES: Dict[int, Type[TypeSafeAPIError]] = {
    400: TypeSafeBadRequestError,
    401: TypeSafeAuthenticationError,
    403: TypeSafePermissionDeniedError,
    404: TypeSafeNotFoundError,
    422: TypeSafeUnprocessableEntityError,
    429: TypeSafeRateLimitError,
}


def api_error(status: int, body: Any, headers: Mapping[str, str], endpoint: Optional[str] = None) -> TypeSafeAPIError:
    t = STATUS_ERROR_TYPES.get(status, TypeSafeInternalServerError if status >= 500 else TypeSafeAPIError)
    return t(status, body, headers, endpoint=endpoint)


# Retries


def _check_timeout(timeout: Any) -> float:
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise TypeSafeError("timeout must be a positive, finite number of seconds.")
    return float(timeout)


def _backoff(attempt: int, initial: float, maximum: float, jitter: float) -> float:
    if initial == 0 or maximum == 0:
        return 0.0
    exponent = attempt - 1
    exponential = maximum if exponent >= math.log2(maximum) - math.log2(initial) else math.ldexp(initial, exponent)
    delay = exponential * (1 - random.random() * jitter)
    return min(exponential, round(delay, 3))


@dataclass(frozen=True)
class RetryPolicy:
    """How a call is retried. The defaults are `typesafe_sdk`'s: two retries after the first
    attempt, backoff from 0.5 s doubling up to 5 s less up to 25% jitter, on 408, 429, any 5xx,
    connection errors and timeouts, honouring `retry-after-ms` and `retry-after`, within 30 s."""

    max_retries: int = 2
    backoff_initial: float = 0.5
    backoff_max: float = 5.0
    backoff_jitter: float = 0.25
    http_statuses: Set[int] = field(default_factory=lambda: {408, 429, *range(500, 600)})
    respect_retry_after: bool = True
    api_connection_error: bool = True
    api_timeout_error: bool = True
    exceptions: Set[Type[BaseException]] = field(default_factory=set)
    predicate: Optional[Callable[[BaseException], bool]] = None
    timeout: Optional[float] = 30.0

    def __post_init__(self) -> None:
        if not isinstance(self.max_retries, int) or self.max_retries < 0:
            raise TypeSafeError("max_retries must be a non-negative integer.")
        for name, value in (("backoff_initial", self.backoff_initial), ("backoff_max", self.backoff_max)):
            if not math.isfinite(value) or value < 0:
                raise TypeSafeError("%s must be a non-negative, finite number of seconds." % name)
        if not 0 <= self.backoff_jitter <= 1:
            raise TypeSafeError("backoff_jitter must be between zero and one.")
        if self.timeout is not None:
            _check_timeout(self.timeout)

    def _retryable(self, error: BaseException) -> bool:
        if isinstance(error, TypeSafeAPITimeoutError):
            builtin = self.api_timeout_error
        elif isinstance(error, TypeSafeAPIConnectionError):
            builtin = self.api_connection_error
        elif isinstance(error, TypeSafeAPIError):
            builtin = error.status in self.http_statuses
        else:
            builtin = False
        return builtin or isinstance(error, tuple(self.exceptions)) or (self.predicate is not None and self.predicate(error))

    def _wait(self, attempt: int, error: BaseException) -> float:
        if self.respect_retry_after and isinstance(error, TypeSafeAPIError):
            delay = parse_retry_after(error.headers)
            if delay is not None:
                return delay / 1000
        return _backoff(attempt, self.backoff_initial, self.backoff_max, self.backoff_jitter)

    def _next(self, attempt: int, error: BaseException, started: float) -> Optional[float]:
        """The wait before attempt `attempt + 1`, or None to give up and raise `error`."""
        if not self._retryable(error) or attempt >= self.max_retries + 1:
            return None
        delay = self._wait(attempt, error)
        if self.timeout is not None and time.monotonic() - started + delay >= self.timeout:
            return None
        return delay


# Questions


def _encode(value: Any) -> Any:
    if isinstance(value, _Question):
        return value.model_dump()
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, Mapping):
        return dict(value)
    if isinstance(value, (tuple, set, frozenset)) or (isinstance(value, Sequence) and not isinstance(value, (str, bytes))):
        return list(value)
    raise TypeError("Encoding objects of type %s is unsupported" % type(value).__name__)


class _Question:
    _fields: Tuple[str, ...] = ("type", "instructions", "criteria")
    _required: Tuple[str, ...] = ()
    _type = ""

    def __init__(self, **kwargs: Any) -> None:
        unknown = sorted(set(kwargs) - set(self._fields))
        if unknown:
            raise ValueError("%s: extra inputs are not permitted: %s" % (type(self).__name__, ", ".join(unknown)))
        missing = [f for f in self._required if f not in kwargs]
        if missing:
            raise ValueError("%s: field required: %s" % (type(self).__name__, ", ".join(missing)))
        t = kwargs.pop("type", self._type)
        if t != self._type:
            raise ValueError("%s: type must be %r" % (type(self).__name__, self._type))
        object.__setattr__(self, "type", t)
        object.__setattr__(self, "instructions", kwargs.pop("instructions", None))
        object.__setattr__(self, "criteria", self._check(kwargs.pop("criteria", None)))

    def _check(self, criteria: Any) -> Any:
        return criteria

    def model_dump(self) -> Dict[str, Any]:
        """The wire form, without the optional fields left at None."""
        return {k: getattr(self, k) for k in self._fields if getattr(self, k) is not None}

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and other.model_dump() == self.model_dump()

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%r" % kv for kv in self.model_dump().items()))


class Noul(_Question):
    """A yes/no question with optional descriptions for either outcome."""

    _type = "noul"

    def _check(self, criteria: Any) -> Any:
        if criteria is None:
            return None
        if not isinstance(criteria, Mapping) or set(criteria) - {"true", "false"}:
            raise ValueError("Noul: criteria must be a mapping with only the keys 'true' and 'false'")
        return dict(criteria)


class Choice(_Question):
    """A question that selects between named alternatives."""

    _type = "choice"
    _required = ("criteria",)

    def _check(self, criteria: Any) -> Any:
        if not isinstance(criteria, Mapping):
            raise ValueError("Choice: criteria must be a mapping of labels to descriptions")
        return dict(criteria)


class Score(_Question):
    """A question that assigns a score using an ordered rubric."""

    _type = "score"
    _required = ("criteria",)

    def _check(self, criteria: Any) -> Any:
        if isinstance(criteria, (str, bytes, Mapping)) or not isinstance(criteria, Sequence) or not criteria:
            raise ValueError("Score: criteria must be a nonempty list of descriptions")
        return list(criteria)


Question = Union[Noul, Choice, Score, Dict[str, Any]]
Questions = Mapping[str, Question]
NoulCriteria = Dict[str, Any]
NoulModel = ChoiceModel = ScoreModel = QuestionModel = Dict[str, Any]


def normalize_questions(questions: Mapping[str, Question]) -> Dict[str, Question]:
    if not questions:
        raise TypeSafeError("At least one question is required.")
    for name, q in questions.items():
        if isinstance(q, Score):
            _score_criteria(name, q.criteria)
        elif not isinstance(q, (Noul, Choice)):
            if not isinstance(q, dict) or not isinstance(q.get("type"), str) or not q["type"]:
                raise TypeSafeError('Question "%s" must be a question object or a dictionary with a nonempty string "type".' % name)
            if q["type"] in ("choice", "score") and "criteria" not in q:
                raise TypeSafeError('Question "%s" requires "criteria".' % name)
            if q["type"] == "score":
                _score_criteria(name, q["criteria"])
    return dict(questions)


def _score_criteria(name: str, criteria: Any) -> None:
    if not criteria:
        raise TypeSafeError('Score question "%s" has no criteria; at least one score is required.' % name)


# Answers


class _Invalid(Exception):
    def __init__(self, path: List[Union[str, int]]):
        super().__init__(path)
        self.path = path


def _format_path(segments: Sequence[Union[str, int]]) -> str:
    path = ""
    for s in segments:
        if isinstance(s, int):
            path += "[%d]" % s
        else:
            path += ".%s" % s if path else str(s)
    return path


def _number(v: Any, path: List[Union[str, int]]) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        raise _Invalid(path)
    return float(v)


def _prob_map(v: Any, path: List[Union[str, int]], int_keys: bool) -> Dict[Any, float]:
    if not isinstance(v, dict):
        raise _Invalid(path)
    out = {}
    for k, p in v.items():
        key: Any = k
        if int_keys:
            try:
                key = int(k)
            except ValueError:
                raise _Invalid(path + [k]) from None
        out[key] = _number(p, path + [k])
    return out


class _Answer:
    _fields: Tuple[str, ...] = ()

    def __setattr__(self, name: str, value: Any) -> None:
        raise AttributeError("%s is frozen" % type(self).__name__)

    def model_dump(self) -> Dict[str, Any]:
        return {k: getattr(self, k) for k in self._fields}

    def __eq__(self, other: Any) -> bool:
        return type(other) is type(self) and other.model_dump() == self.model_dump()

    def __hash__(self) -> int:
        return hash((type(self), repr(self.model_dump())))

    def __repr__(self) -> str:
        return "%s(%s)" % (type(self).__name__, ", ".join("%s=%r" % kv for kv in self.model_dump().items()))


class NoulAnswer(_Answer):
    """A yes/no answer: `noul` is the probability of yes."""

    _fields = ("type", "noul")

    def __init__(self, raw: Dict[str, Any], path: List[Union[str, int]]):
        object.__setattr__(self, "type", "noul")
        object.__setattr__(self, "noul", _number(raw.get("noul"), path + ["noul"]))


class ChoiceAnswer(_Answer):
    """A selected label and its probabilities."""

    _fields = ("type", "choice", "confidence", "probabilities")

    def __init__(self, raw: Dict[str, Any], path: List[Union[str, int]]):
        if not isinstance(raw.get("choice"), str):
            raise _Invalid(path + ["choice"])
        object.__setattr__(self, "type", "choice")
        object.__setattr__(self, "choice", raw["choice"])
        object.__setattr__(self, "confidence", _number(raw.get("confidence"), path + ["confidence"]))
        object.__setattr__(self, "probabilities", _prob_map(raw.get("probabilities"), path + ["probabilities"], False))


class ScoreAnswer(_Answer):
    """An expected score with its rubric and probabilities, keyed by integer level."""

    _fields = ("type", "score", "confidence", "legend", "probabilities")

    def __init__(self, raw: Dict[str, Any], path: List[Union[str, int]]):
        object.__setattr__(self, "type", "score")
        object.__setattr__(self, "score", _number(raw.get("score"), path + ["score"]))
        object.__setattr__(self, "confidence", _number(raw.get("confidence"), path + ["confidence"]))
        legend = raw.get("legend")
        if not isinstance(legend, dict):
            raise _Invalid(path + ["legend"])
        out = {}
        for k, v in legend.items():
            try:
                out[int(k)] = v
            except ValueError:
                raise _Invalid(path + ["legend", k]) from None
            if not isinstance(v, (str, dict, list)):
                raise _Invalid(path + ["legend", k])
        object.__setattr__(self, "legend", out)
        object.__setattr__(self, "probabilities", _prob_map(raw.get("probabilities"), path + ["probabilities"], True))


Answer = Union[NoulAnswer, ChoiceAnswer, ScoreAnswer]
_ANSWERS = {"noul": NoulAnswer, "choice": ChoiceAnswer, "score": ScoreAnswer}


class Usage(_Answer):
    """Token counts for a request, when the server reports them."""

    _fields = ("input_tokens", "output_tokens")

    def __init__(self, raw: Any, path: List[Union[str, int]]):
        if not isinstance(raw, dict):
            raise _Invalid(path)
        for k in self._fields:
            v = raw.get(k)
            if v is not None and (isinstance(v, bool) or not isinstance(v, int)):
                raise _Invalid(path + [k])
            object.__setattr__(self, k, v)


class RawResponse:
    """The HTTP response a result was read from: `status_code`, `headers` and `content`."""

    def __init__(self, method: str, url: str, status: int, headers: Headers, content: bytes):
        self.method, self.url = method, url
        self.status_code = status
        self.headers = headers
        self.content = content

    @property
    def is_success(self) -> bool:
        return 200 <= self.status_code < 300

    @property
    def text(self) -> str:
        return self.content.decode("utf-8", errors="replace")

    def json(self) -> Any:
        return json.loads(self.content)

    def __repr__(self) -> str:
        return "<RawResponse [%d]>" % self.status_code


class _Response:
    _raw: Optional[RawResponse] = None

    @property
    def request_id(self) -> str:
        rid = self._raw.headers.get(REQUEST_ID_HEADER) if self._raw is not None else None
        if rid is None:
            raise TypeSafeError("The response did not include a request ID.")
        return rid

    @property
    def raw_http_response(self) -> RawResponse:
        if self._raw is None:
            raise TypeSafeError("The response was not created from a raw HTTP response.")
        return self._raw


class SystemOneResponse(_Response):
    """Answers keyed by question name, with `nouls`, `choices` and `scores` by type."""

    def __init__(self, decoded: Dict[str, Any]):
        if not isinstance(decoded.get("model"), str):
            raise _Invalid(["model"])
        self.model: str = decoded["model"]
        self.usage = Usage(decoded.get("usage"), ["usage"])
        answers = decoded.get("answers", {})
        if not isinstance(answers, dict):
            raise _Invalid(["answers"])
        self.answers: Dict[str, Answer] = {
            name: _ANSWERS[raw["type"]](raw, ["answers", name]) for name, raw in answers.items()
        }
        self.nouls: Dict[str, NoulAnswer] = {k: a for k, a in self.answers.items() if isinstance(a, NoulAnswer)}
        self.choices: Dict[str, ChoiceAnswer] = {k: a for k, a in self.answers.items() if isinstance(a, ChoiceAnswer)}
        self.scores: Dict[str, ScoreAnswer] = {k: a for k, a in self.answers.items() if isinstance(a, ScoreAnswer)}

    def model_dump(self) -> Dict[str, Any]:
        return {"model": self.model, "usage": self.usage.model_dump(),
                "answers": {k: a.model_dump() for k, a in self.answers.items()}}

    def __repr__(self) -> str:
        return "SystemOneResponse(model=%r, answers=%r, usage=%r)" % (self.model, self.answers, self.usage)


class ModelMetadata(_Answer):
    """One model: `name`, `description` and `release_date` (YYYY-MM-DD)."""

    _fields = ("name", "description", "release_date")

    def __init__(self, raw: Any, path: List[Union[str, int]]):
        if not isinstance(raw, dict):
            raise _Invalid(path)
        for k in self._fields:
            if not isinstance(raw.get(k), str):
                raise _Invalid(path + [k])
            object.__setattr__(self, k, raw[k])


class ListModelsResponse(_Response):
    """The models the server offers."""

    def __init__(self, decoded: Any):
        if not isinstance(decoded, dict) or not isinstance(decoded.get("models"), list):
            raise _Invalid(["models"])
        self.models: Tuple[ModelMetadata, ...] = tuple(
            ModelMetadata(m, ["models", i]) for i, m in enumerate(decoded["models"])
        )

    def __repr__(self) -> str:
        return "ListModelsResponse(models=%r)" % (self.models,)


def _deserialize(content: bytes) -> Any:
    if not content:
        return None
    try:
        return json.loads(content)
    except ValueError:
        return content.decode("utf-8", errors="replace")


def _parse(raw: RawResponse, response_type: Any) -> Any:
    endpoint = "%s %s" % (raw.method, raw.url.split("?", 1)[0])
    body = _deserialize(raw.content)
    if not raw.is_success:
        raise api_error(raw.status_code, body, raw.headers, endpoint)

    def invalid(path: str) -> TypeSafeAPIResponseValidationError:
        return TypeSafeAPIResponseValidationError(raw.status_code, body, raw.headers, path, endpoint)

    if response_type is ListModelsResponse:
        try:
            out = ListModelsResponse(body)
        except _Invalid as e:
            raise invalid(_format_path(e.path)) from None
    else:
        if not isinstance(body, dict):
            raise invalid("")
        answers = body.get("answers")
        if isinstance(answers, dict):
            for name, a in list(answers.items()):
                if not isinstance(a, dict) or not isinstance(a.get("type"), str):
                    raise invalid("answers.%s.type" % name)
                if a["type"] not in _ANSWERS:
                    logger.warning("Ignoring answer %r with unrecognized type %r", name, a["type"])
                    del answers[name]
        if response_type is SystemOneResponse:
            try:
                out = SystemOneResponse(body)
            except _Invalid as e:
                raise invalid(_format_path(e.path)) from None
        else:
            # A Pydantic model: the answers it declares as fields are lifted to the top level.
            fields = set(getattr(response_type, "model_fields", {})) - {"model", "usage", "answers"}
            if isinstance(answers, dict):
                for f in fields & set(answers):
                    body[f] = answers[f]
            try:
                out = response_type.model_validate(body)
            except Exception as e:
                errors = getattr(e, "errors", None)
                if errors is None:
                    raise
                raise invalid(_format_path([s for s in errors(include_url=False)[0]["loc"] if s != "[key]"])) from e
            try:
                object.__setattr__(out, "_kime_raw", raw)
            except Exception:
                pass
            return out
    out._raw = raw
    return out


# Transports


class _HTTPTransport:
    """HTTP/1.1 over one kept-alive connection per thread."""

    def __init__(self) -> None:
        self._local = threading.local()

    def _conn(self, url: str, timeout: float) -> Tuple[http.client.HTTPConnection, str, bool]:
        u = urlsplit(url)
        key = (u.scheme, u.netloc)
        c = getattr(self._local, "conns", None)
        if c is None:
            c = self._local.conns = {}
        path = (u.path or "/") + ("?" + u.query if u.query else "")
        if key in c:
            conn = c[key]
            conn.timeout = timeout
            if conn.sock is not None:
                conn.sock.settimeout(timeout)
            return conn, path, True
        cls = http.client.HTTPSConnection if u.scheme == "https" else http.client.HTTPConnection
        conn = c[key] = cls(u.hostname or "127.0.0.1", u.port, timeout=timeout)
        return conn, path, False

    def _drop(self, url: str) -> None:
        u = urlsplit(url)
        conn = getattr(self._local, "conns", {}).pop((u.scheme, u.netloc), None)
        if conn is not None:
            conn.close()

    def handle(self, method: str, url: str, headers: Headers, content: Optional[bytes], timeout: float
               ) -> Tuple[int, Headers, bytes]:
        for tries in (0, 1):
            conn, path, reused = self._conn(url, timeout)
            try:
                conn.request(method, path, body=content, headers=dict(headers))
                r = conn.getresponse()
                body = r.read()
                out = Headers(r.getheaders())
                if (out.get("connection") or "").lower() == "close":
                    self._drop(url)
                return r.status, out, body
            except (http.client.RemoteDisconnected, BrokenPipeError, ConnectionResetError):
                self._drop(url)
                # A kept-alive connection the server has closed is tried once more on a new one.
                if reused and tries == 0:
                    continue
                raise
            except BaseException:
                self._drop(url)
                raise
        raise AssertionError("unreachable")

    def close(self) -> None:
        for conn in getattr(self._local, "conns", {}).values():
            conn.close()
        self._local = threading.local()


class _LocalTransport:
    """Answers `POST /v1/systemone` and `GET /v1/models` in process, as `kime serve` would."""

    def __init__(self, local: Any, **engine: Any):
        from . import Agent

        self.agent = local if isinstance(local, Agent) else Agent(local, **engine)

    def handle(self, method: str, url: str, headers: Headers, content: Optional[bytes], timeout: float
               ) -> Tuple[int, Headers, bytes]:
        path = urlsplit(url).path
        rid = "req_" + uuid.uuid4().hex
        out = Headers({"content-type": "application/json", "x-request-id": rid, REQUEST_ID_HEADER: rid})
        engine = self.agent._engine
        if method == "POST" and path.endswith(SYSTEM_ONE_PATH):
            status, body = engine.decide_jev((content or b"").decode())
            return status, out, body.encode()
        if method == "GET" and path.endswith(MODELS_PATH):
            from . import _native

            models = [{"name": n, "description": d, "release_date": _native.RELEASE_DATE} for n, d in (
                ("kime-latest", "Default model. Served by %s." % engine.model_id),
                (engine.model_id, "%s on %s." % (engine.model_id, engine.device)),
                ("jev-latest", "Alias. Served by kime-latest."),
            )]
            return 200, out, _dumps({"models": models})
        return 404, out, b'{"detail":"Not Found"}'

    def close(self) -> None:
        pass


# Clients


@dataclass
class _Config:
    api_key: Optional[str] = field(repr=False)
    base_url: str
    default_model: str
    timeout: float
    default_headers: Headers = field(repr=False)


def _resolve(api_key: Optional[str], base_url: Optional[str], model: Optional[str], timeout: Optional[float],
             headers: Optional[Mapping[str, str]], local: bool) -> _Config:
    url = (base_url if base_url is not None else _env(BASE_URL_ENV) or DEFAULT_BASE_URL).rstrip("/")
    if local:
        url = "http://local"
    key = (api_key if api_key is not None else _env(API_KEY_ENV)).strip()
    if key and (not key.isascii() or not key.isprintable() or " " in key):
        raise TypeSafeError("API key must contain only printable ASCII characters without whitespace.")
    if not key and url == TYPESAFE_BASE_URL:
        raise TypeSafeError("No API key was provided. Pass api_key or set the %s environment variable." % API_KEY_ENV)
    return _Config(
        key or None,
        url,
        model if model is not None else _env(DEFAULT_MODEL_ENV) or DEFAULT_MODEL,
        _check_timeout(DEFAULT_TIMEOUT if timeout is None else timeout),
        Headers(headers),
    )


@dataclass(frozen=True)
class _Request:
    method: str
    url: str
    headers: Headers
    content: Optional[bytes]
    timeout: float
    response_type: Any


def _prepare(config: _Config, method: str, path: str, body: Any, timeout: Optional[float],
             headers: Optional[Mapping[str, str]], response_type: Any) -> _Request:
    merged = Headers(config.default_headers)
    merged.update(headers or {})
    merged.pop(RETRY_COUNT_HEADER)
    fixed = {"Accept": "application/json", "User-Agent": SDK, "X-TypeSafe-SDK": SDK, "X-TypeSafe-Runtime": RUNTIME}
    if config.api_key:
        fixed["Authorization"] = "Bearer %s" % config.api_key
    merged.update(fixed)
    content = None
    if body is not None:
        try:
            content = _dumps(body)
        except (TypeError, ValueError) as e:
            raise TypeSafeError("The request body could not be encoded as JSON") from e
        merged["Content-Type"] = "application/json"
    return _Request(method, config.base_url + path, merged, content,
                    _check_timeout(config.timeout if timeout is None else timeout), response_type)


def _attempt(transport: Any, req: _Request, attempts: int) -> Any:
    headers = Headers(req.headers)
    if attempts:
        headers[RETRY_COUNT_HEADER] = str(attempts)
        logger.info("%s %s retry %s", req.method, req.url, attempts)
    started = time.monotonic()
    try:
        status, rh, content = transport.handle(req.method, req.url, headers, req.content, req.timeout)
    except (socket.timeout, TimeoutError) as e:
        raise TypeSafeAPITimeoutError(req.timeout) from e
    except (OSError, http.client.HTTPException) as e:
        raise TypeSafeAPIConnectionError("Connection error: %s" % e) from e
    logger.info("%s %s <- %s in %.0fms (request %s)", req.method, req.url, status,
                (time.monotonic() - started) * 1000, rh.get(REQUEST_ID_HEADER, "-"))
    return _parse(RawResponse(req.method, req.url, status, rh, content), req.response_type)


def _send(transport: Any, policy: RetryPolicy, req: _Request) -> Any:
    started = time.monotonic()
    attempt = 0
    while True:
        try:
            return _attempt(transport, req, attempt)
        except Exception as e:
            attempt += 1
            delay = policy._next(attempt, e, started)
            if delay is None:
                raise
            time.sleep(delay)


class Models:
    """The models the server offers, reached through `TypeSafeClient.models`."""

    def __init__(self, client: "TypeSafeClient"):
        self._client = client

    def list(self, *, retry: Optional[RetryPolicy] = None, timeout: Optional[float] = None,
             extra_headers: Optional[Mapping[str, str]] = None) -> ListModelsResponse:
        c = self._client
        req = _prepare(c._config, "GET", MODELS_PATH, None, timeout, extra_headers, ListModelsResponse)
        return _send(c._transport, retry or c._retry, req)


class TypeSafeClient:
    """A System One client with `typesafe_sdk.TypeSafeClient`'s arguments, plus `local=` to answer
    in process. `device`, `threads` and `precision` go to the local model."""

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        retry: Optional[RetryPolicy] = None,
        timeout: Optional[float] = None,
        headers: Optional[Mapping[str, str]] = None,
        transport: Any = None,
        http_client: Any = None,
        base_url: Optional[str] = None,
        local: Any = None,
        device: Optional[str] = None,
        threads: int = 0,
        precision: Optional[str] = None,
    ) -> None:
        if transport is not None and http_client is not None:
            raise ValueError("transport and http_client are mutually exclusive.")
        if http_client is not None:
            raise NotImplementedError("kime has no httpx; pass transport= instead")
        if local is not None and transport is not None:
            raise ValueError("local and transport are mutually exclusive.")
        self._config = _resolve(api_key, base_url, model, timeout, headers, local is not None)
        self._retry = retry or RetryPolicy()
        if local is not None:
            transport = _LocalTransport(local, device=device, threads=threads, precision=precision)
        self._transport = transport if transport is not None else _HTTPTransport()
        self.models = Models(self)

    def system_one(
        self,
        state: JSONContent,
        questions: Mapping[str, Question],
        *,
        model: Optional[str] = None,
        retry: Optional[RetryPolicy] = None,
        timeout: Optional[float] = None,
        extra_headers: Optional[Mapping[str, str]] = None,
        extra_body: Optional[Mapping[str, Any]] = None,
        response_model: Any = None,
    ) -> Any:
        """Answers named questions about text or structured state. `extra_body` is merged over
        the body last, so it can carry the `kime` options."""
        body: Dict[str, Any] = {
            "state": state,
            "model": self._config.default_model if model is None else model,
            "questions": normalize_questions(questions),
        }
        if extra_body is not None:
            body.update(extra_body)
        req = _prepare(self._config, "POST", SYSTEM_ONE_PATH, body, timeout, extra_headers,
                       SystemOneResponse if response_model is None else response_model)
        return _send(self._transport, retry or self._retry, req)

    def close(self) -> None:
        self._transport.close()

    def __enter__(self) -> "TypeSafeClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()


class AsyncModels:
    """`Models` for `AsyncTypeSafeClient`."""

    def __init__(self, client: "AsyncTypeSafeClient"):
        self._client = client

    async def list(self, *, retry: Optional[RetryPolicy] = None, timeout: Optional[float] = None,
                   extra_headers: Optional[Mapping[str, str]] = None) -> ListModelsResponse:
        return await asyncio.to_thread(self._client._sync.models.list, retry=retry, timeout=timeout,
                                       extra_headers=extra_headers)


class AsyncTypeSafeClient:
    """`TypeSafeClient` for asyncio. Each call runs on a worker thread, and a local model runs
    with the GIL released, so the event loop is never blocked."""

    def __init__(self, **kwargs: Any) -> None:
        self._sync = TypeSafeClient(**kwargs)
        self.models = AsyncModels(self)

    async def system_one(self, state: JSONContent, questions: Mapping[str, Question], **kwargs: Any) -> Any:
        return await asyncio.to_thread(self._sync.system_one, state, questions, **kwargs)

    async def close(self) -> None:
        self._sync.close()

    async def __aenter__(self) -> "AsyncTypeSafeClient":
        return self

    async def __aexit__(self, *exc: Any) -> None:
        await self.close()


__all__ = [
    "Answer", "AsyncModels", "AsyncTypeSafeClient", "Choice", "ChoiceAnswer", "ChoiceModel", "Headers",
    "JSONContent", "JSONValue", "ListModelsResponse", "ModelMetadata", "Models", "Noul", "NoulAnswer",
    "NoulCriteria", "NoulModel", "Question", "QuestionModel", "Questions", "RawResponse", "RetryPolicy", "Score",
    "ScoreAnswer", "ScoreModel", "SystemOneResponse", "TypeSafeAPIConnectionError", "TypeSafeAPIError",
    "TypeSafeAPIResponseValidationError", "TypeSafeAPITimeoutError", "TypeSafeAuthenticationError",
    "TypeSafeBadRequestError", "TypeSafeClient", "TypeSafeError", "TypeSafeInternalServerError",
    "TypeSafeNotFoundError", "TypeSafePermissionDeniedError", "TypeSafeRateLimitError",
    "TypeSafeUnprocessableEntityError", "Usage",
]
