"""Laya's embedding shortlist for choice questions with many labels.

This is `laya.shortlist` from Laya 0.3.20 (Apache-2.0). `predict_shortlist` embeds the state and
each option with an `embed_fn` you pass, keeps the `k` closest options of every choice question
and answers once on what is left. The ranking, the ties, the checks and the `shortlist` entry in
the result are Laya's. It needs no numpy: `embed_fn` may return a numpy array, a torch tensor or
a list of lists. `embed_fn_from_agent` gives one that pools the checkpoint's own encoder.
"""

import json
import math
from typing import Any, Callable, Dict, List, Optional, Sequence

DEFAULT_SHORTLIST_K = 20


def shortlist_choice(
    state: Any,
    criteria: Any,
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    *,
    instructions: Optional[str] = None,
) -> List[Any]:
    """The `k` choice labels closest to `state`, closest first.

    `embed_fn` maps a list of strings to an array of shape `(len(texts), dim)`. It is called
    once, with the query first and then one string per option in criteria order, rendered the
    way the model reads them. When `k` is at least the number of labels, every label comes back
    in its own order and `embed_fn` is not called. Ties keep the earlier label.
    """
    labels, _scores, _passthrough, _n = _rank(state, criteria, embed_fn, k, instructions)
    return labels


def predict_shortlist(
    agent: Any,
    state: Any,
    questions: Dict[str, Dict[str, Any]],
    embed_fn: Callable[[Sequence[str]], Any],
    k: int = DEFAULT_SHORTLIST_K,
    **predict_kwargs: Any,
) -> Dict[str, Any]:
    """Shortlists each choice question, then calls `predict` or `system_one` once.

    Other questions, and choices with at most `k` labels, go through unchanged. `questions` is
    not changed. The result is the model's plus a `shortlist` entry, where `shortlist[qid]` has
    `labels` (closest first), `scores` (the cosines, or None when nothing was dropped), `k`,
    `n` and `passthrough`. Probabilities on a shortlisted choice are over the kept labels only.
    Extra keyword arguments go to `predict` or `system_one`, such as `model=` on a `Router`.
    """
    if not isinstance(questions, dict):
        raise TypeError("questions must be a dict of question id -> definition")
    checked = _check_k(k)
    reduced: Dict[str, Any] = {}
    meta: Dict[str, Dict[str, Any]] = {}
    for qid, qdef in questions.items():
        if not isinstance(qdef, dict) or qdef.get("type") != "choice":
            reduced[qid] = qdef
            continue
        if "criteria" not in qdef:
            raise ValueError("question %r is a choice but has no criteria" % (qid,))
        labels, scores, passthrough, n = _rank(state, qdef["criteria"], embed_fn, checked, qdef.get("instructions"))
        meta[qid] = {"labels": list(labels), "scores": scores, "k": checked, "n": n, "passthrough": passthrough}
        if passthrough:
            reduced[qid] = qdef
            continue
        updated = dict(qdef)
        updated["criteria"] = _subset_criteria(qdef["criteria"], labels)
        reduced[qid] = updated

    result = _call_predict(agent, state, reduced, **predict_kwargs)
    if not isinstance(result, dict):
        raise TypeError("predict/system_one must return a dict, got %s" % type(result).__name__)
    out = dict(result)
    out["shortlist"] = meta
    return out


def embed_fn_from_agent(agent: Any, max_length: int = 512, batch_size: int = 32):
    """Mean pools the encoder of the checkpoint `agent` already has loaded.

    The callable embeds a list of strings with the checkpoint's own tokenizer and encoder and
    returns a list of rows as wide as the encoder. It runs no decision head and loads nothing.
    Each text is cut to `max_length` tokens with the specials, and `batch_size` bounds how many
    texts go to the engine at once, as in Laya. A dedicated bi-encoder passed as `embed_fn` will
    usually shortlist better.
    """
    if isinstance(max_length, bool) or not isinstance(max_length, int) or max_length < 1:
        raise ValueError("max_length must be a positive integer, got %r" % (max_length,))
    if isinstance(batch_size, bool) or not isinstance(batch_size, int) or batch_size < 1:
        raise ValueError("batch_size must be a positive integer, got %r" % (batch_size,))

    def embed_fn(texts: Sequence[str]) -> List[List[float]]:
        rows = list(texts)
        out: List[List[float]] = []
        for start in range(0, len(rows), batch_size):
            out.extend(agent.embed(rows[start : start + batch_size], max_length))
        return out

    return embed_fn


def _rank(state, criteria, embed_fn, k, instructions):
    checked = _check_k(k)
    items = _criteria_items(criteria)
    n = len(items)
    keys = [key for key, _value in items]
    if checked >= n:
        return list(keys), None, True, n
    query = _query_text(state, instructions)
    matrix = _embeddings(embed_fn, [query] + _option_texts(items))
    sims = _cosine(matrix[0], matrix[1:])
    order = sorted(range(n), key=lambda i: -sims[i])[:checked]
    return [keys[i] for i in order], [sims[i] for i in order], False, n


def _check_k(k: int) -> int:
    if isinstance(k, bool) or not isinstance(k, int) or k < 1:
        raise ValueError("k must be a positive integer, got %r" % (k,))
    return k


def _criteria_items(criteria):
    if isinstance(criteria, dict):
        items = list(criteria.items())
    elif isinstance(criteria, list):
        items = [(item, None) for item in criteria]
    else:
        raise TypeError("choice criteria must be a dict or list, got %s" % type(criteria).__name__)
    if not items:
        raise ValueError("choice criteria must contain at least one option")
    seen = set()
    for key, _value in items:
        if key in seen:
            raise ValueError("choice criteria label %r is duplicated" % (key,))
        seen.add(key)
    return items


def _criterion(value) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def _option_texts(items) -> List[str]:
    # Laya's render_options for a choice: None and "" mean no description.
    return [str(k) if v is None or v == "" else "%s: %s" % (k, _criterion(v)) for k, v in items]


def _query_text(state, instructions) -> str:
    body = state if isinstance(state, str) else json.dumps(state, ensure_ascii=False)
    if instructions is None or instructions == "":
        return body
    if not isinstance(instructions, str):
        instructions = json.dumps(instructions, ensure_ascii=False)
    return "%s\n%s" % (instructions, body)


def _subset_criteria(criteria, labels):
    if isinstance(criteria, dict):
        return {label: criteria[label] for label in labels}
    return list(labels)


def _finite(x: Any) -> float:
    x = float(x)
    return x if math.isfinite(x) else 0.0


def _embeddings(embed_fn, texts: Sequence[str]) -> List[List[float]]:
    if not callable(embed_fn):
        raise TypeError("embed_fn must be callable")
    raw = embed_fn(list(texts))
    if hasattr(raw, "detach"):
        raw = raw.detach().float().cpu()
    if hasattr(raw, "tolist"):
        raw = raw.tolist()
    try:
        rows = [[_finite(x) for x in row] for row in raw]
        shape = (len(rows), len(rows[0]) if rows else 0)
        ragged = any(len(r) != shape[1] for r in rows)
    except TypeError:
        rows, shape, ragged = [], (len(raw),) if hasattr(raw, "__len__") else (), True
    if ragged or shape[0] != len(texts) or shape[1] < 1:
        raise ValueError("embed_fn must return an array of shape (%d, dim), got %s" % (len(texts), shape))
    return rows


def _cosine(query: List[float], docs: List[List[float]]) -> List[float]:
    qn = math.sqrt(math.fsum(x * x for x in query))
    if qn == 0.0 or not docs:
        return [0.0] * len(docs)
    out = []
    for d in docs:
        denom = math.sqrt(math.fsum(x * x for x in d)) * qn
        out.append(max(-1.0, min(1.0, math.fsum(a * b for a, b in zip(d, query)) / denom)) if denom > 0 else 0.0)
    return out


def _call_predict(agent, state, questions, **predict_kwargs):
    fn = getattr(agent, "predict", None)
    if fn is None:
        fn = getattr(agent, "system_one", None)
    if fn is None:
        raise TypeError("agent must provide predict or system_one")
    return fn(state, questions, **predict_kwargs)


__all__ = ["DEFAULT_SHORTLIST_K", "embed_fn_from_agent", "predict_shortlist", "shortlist_choice"]
