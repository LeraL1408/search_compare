from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from .common import AFLOW_BASELINE, BASELINE, append, save


class BudgetStop(BaseException):
    """Escapes upstream broad Exception handlers; no unpaid retry loops."""


@dataclass
class Ledger:
    limits: dict
    path: object
    calls: int = 0
    tokens: int = 0
    usd: float = 0.0
    uncertain: bool = False
    cost_available: bool = True
    token_usage_complete: bool = True
    input_tokens: int = 0
    output_tokens: int = 0
    cached_input_tokens: int = 0
    failures: int = 0
    codex_runs: int = 0
    started: float = field(default_factory=time.monotonic)
    parent: Ledger | None = None

    def check(self, tokens=0, usd=0):
        if self.parent:
            self.parent.check(tokens, usd)
        if self.parent is None and not self.token_usage_complete:
            raise BudgetStop("Total token usage is incomplete; stopping further calls")
        for key, value in (("max_calls", self.calls + 1), ("max_tokens", self.tokens + tokens), ("max_usd", self.usd + usd), ("max_seconds", time.monotonic() - self.started)):
            if self.limits[key] is not None and value > self.limits[key]:
                raise BudgetStop(f"Reached {key}")

    def ancestors(self):
        node = self.parent
        while node:
            yield node
            node = node.parent

    def counters(self):
        return {key: getattr(self, key) for key in ("calls", "tokens", "usd", "input_tokens", "output_tokens", "cached_input_tokens", "failures", "codex_runs")}

    def propagate(self, before):
        for parent in self.ancestors():
            for key, value in self.counters().items():
                setattr(parent, key, getattr(parent, key) + value - before[key])
            parent.cost_available &= self.cost_available
            parent.token_usage_complete &= self.token_usage_complete
            parent.uncertain |= self.uncertain
            save(parent.path, parent.snapshot())

    def snapshot(self):
        return {"calls": self.calls, "tokens": self.tokens, "usd": self.usd if self.cost_available else None,
                "cost_uncertain": self.uncertain, "seconds": time.monotonic() - self.started,
                "cost_available": self.cost_available, "token_usage_complete": self.token_usage_complete,
                "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
                "cached_input_tokens": self.cached_input_tokens, "failures": self.failures,
                "codex_runs": self.codex_runs,
                "limits": self.limits}


class Gateway:
    def __init__(self, models, ledger, seed, fixture=False):
        self.models, self.ledger, self.seed, self.fixture = models, ledger, seed, fixture
        self.clients = {}
        if not fixture and any(m.get("provider") == "codex" for m in models.values()):
            self.ledger.cost_available = False
            for parent in self.ledger.ancestors():
                parent.cost_available = False

    def call(self, messages, role="execution", temperature=0, tools=None, fields=None):
        self.ledger.check()
        # Codex reports complete usage only after a turn. Leave a conservative
        # payload-byte + runtime/output reserve before launching near the total
        # cap; the provider still has no server-enforced output-token ceiling.
        ancestors = list(self.ledger.ancestors())
        if ancestors and not self.fixture and self.models[role].get("provider") == "codex":
            reserve = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()) + 16000
            ancestors[-1].check(tokens=reserve)
        before = self.ledger.counters()
        try:
            result = self._call(messages, role, temperature, tools, fields)
        finally:
            self.ledger.propagate(before)
        for parent in ancestors:
            if parent.tokens > parent.limits["max_tokens"]:
                raise BudgetStop("Shared token budget reached by the last Codex invocation")
        return result

    def _call(self, messages, role="execution", temperature=0, tools=None, fields=None):
        config = self.models[role]
        if not self.fixture and config.get("provider") == "codex":
            return self._call_codex(messages, role, temperature, tools, fields)
        # UTF-8 bytes plus protocol allowance is deliberately conservative. We do
        # not truncate prompts: large archives stop before spending beyond limits.
        upper_input = len(json.dumps({"messages": messages, "tools": tools}, ensure_ascii=False).encode()) + 1024
        output_cap = config["max_output_tokens"]
        if upper_input > config["max_input_tokens"]:
            raise BudgetStop("Request exceeds configured input envelope")
        reservation = (upper_input * config["input_usd_per_million"] + output_cap * config["output_usd_per_million"]) / 1e6
        self.ledger.check(upper_input + output_cap, reservation)
        started = time.monotonic()
        self.ledger.calls += 1
        try:
            if self.fixture:
                message = self._fixture(messages, role, fields)
                usage = {"prompt_tokens": max(1, upper_input // 4), "completion_tokens": max(1, len(str(message)) // 4)}
            else:
                from openai import OpenAI
                if role not in self.clients:
                    key = os.getenv(config.get("api_key_env", "OPENAI_API_KEY"))
                    if not key:
                        raise ValueError("Model API key is missing")
                    self.clients[role] = OpenAI(api_key=key, base_url=config["base_url"], max_retries=0)
                kwargs = {"model": config["model"], "messages": messages,
                          config.get("output_limit_parameter", "max_completion_tokens"): output_cap}
                if config.get("send_temperature", True):
                    kwargs["temperature"] = temperature
                if config.get("send_seed", False):
                    kwargs["seed"] = self.seed + self.ledger.calls
                if tools:
                    kwargs["tools"] = tools
                if fields:
                    kwargs["response_format"] = {"type": "json_object"}
                remaining = self.ledger.limits["max_seconds"] - (time.monotonic() - self.ledger.started)
                response = self.clients[role].chat.completions.create(**kwargs, timeout=min(config.get("timeout_seconds", 90), max(0.01, remaining)))
                message = response.choices[0].message.model_dump(exclude_none=True)
                usage = response.usage.model_dump() if response.usage else None
            if not usage or not all(isinstance(usage.get(k), int) and usage[k] >= 0 for k in ("prompt_tokens", "completion_tokens")):
                raise ValueError("Provider omitted token usage; charged reservation and stopped")
        except BaseException as exc:
            self.ledger.tokens += upper_input + output_cap
            self.ledger.usd += reservation
            self.ledger.uncertain = True
            self.ledger.failures += 1
            self.ledger.token_usage_complete = False
            append(self.ledger.path, {"role": role, "status": "uncertain", "error": type(exc).__name__,
                                      "reserved_usd": reservation, "budget": self.ledger.snapshot()})
            raise BudgetStop("Provider failed; spend is uncertain, stopping this phase") from exc
        actual_usd = (usage["prompt_tokens"] * config["input_usd_per_million"] + usage["completion_tokens"] * config["output_usd_per_million"]) / 1e6
        self.ledger.tokens += usage["prompt_tokens"] + usage["completion_tokens"]
        self.ledger.input_tokens += usage["prompt_tokens"]
        self.ledger.output_tokens += usage["completion_tokens"]
        self.ledger.cached_input_tokens += (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
        self.ledger.usd += actual_usd
        append(self.ledger.path, {"role": role, "model": config["model"], "usage": usage, "usd": actual_usd,
                                  "seconds": time.monotonic() - started, "messages": messages,
                                  "response": message, "budget": self.ledger.snapshot()})
        if usage["prompt_tokens"] > upper_input or usage["completion_tokens"] > output_cap:
            self.ledger.uncertain = True
            raise BudgetStop("Provider exceeded token envelope; stop and review budget assumptions")
        return message

    def _call_codex(self, messages, role, temperature, tools, fields):
        from .codex_backend import invoke
        self.ledger.check()
        self.ledger.calls += 1
        config = self.models[role]
        started = time.monotonic()
        event_path = self.ledger.path.parent / "codex_events" / f"{self.ledger.path.stem}-{self.ledger.calls:05}.jsonl"
        remaining = self.ledger.limits["max_seconds"] - (started - self.ledger.started)
        for parent in self.ledger.ancestors():
            remaining = min(remaining, parent.limits["max_seconds"] - (started - parent.started))
        def started_process():
            self.ledger.codex_runs += 1
        try:
            message, usage = invoke(config, messages, tools, fields,
                                    min(config.get("timeout_seconds", 120), max(0.01, remaining)), event_path,
                                    on_start=started_process)
        except BaseException as exc:
            usage = getattr(exc, "usage", None)
            if usage:
                self._record_codex_usage(usage)
            else:
                self.ledger.token_usage_complete = False
            self.ledger.failures += 1
            append(self.ledger.path, {"provider": "codex", "role": role, "model": config["model"],
                                     "status": "failed", "error": type(exc).__name__ + ": " + str(exc),
                                     "usage": usage, "usd": None, "events": str(event_path),
                                     "seconds": time.monotonic() - started, "budget": self.ledger.snapshot()})
            if isinstance(exc, (KeyboardInterrupt, SystemExit)):
                raise
            raise BudgetStop("Codex invocation failed; stopped phase to preserve accounting: " + str(exc)) from exc
        self._record_codex_usage(usage)
        for index, call in enumerate(message.get("tool_calls") or []):
            call["id"] = f"codex-{self.ledger.calls}-{index}"
        append(self.ledger.path, {"provider": "codex", "role": role, "model": config["model"],
                                 "status": "complete", "usage": usage, "usd": None,
                                 "messages": messages, "response": message, "events": str(event_path),
                                 "seconds": time.monotonic() - started, "temperature_applied": False,
                                 "seed_applied": False, "budget": self.ledger.snapshot()})
        if self.ledger.tokens > self.ledger.limits["max_tokens"]:
            raise BudgetStop("Codex token limit exceeded by the last invocation; token caps are checked between runs")
        return message

    def _record_codex_usage(self, usage):
        self.ledger.input_tokens += usage["prompt_tokens"]
        self.ledger.output_tokens += usage["completion_tokens"]
        self.ledger.cached_input_tokens += usage.get("cached_input_tokens", 0)
        self.ledger.tokens += usage["prompt_tokens"] + usage["completion_tokens"]

    def _fixture(self, messages, role, fields):
        """Deterministic plumbing checks, not model or benchmark performance."""
        text = "\n".join(str(m.get("content", "")) for m in messages)
        if role == "search":
            if "<graph>" in text:
                result = ("<modification>fixture " + str(self.ledger.calls) + "</modification><graph>" +
                          AFLOW_BASELINE["code"] + "</graph><prompt># fixture</prompt>")
            else:
                result = json.dumps({"thought": "offline fixture", "name": "Fixture candidate",
                                     "code": BASELINE["code"]})
        else:
            match = re.search(r"Fixture question: (\d+) plus (\d+)", text)
            value = str(int(match[1]) + int(match[2])) if match else "fixture"
            if "Fixture code:" in text:
                value = "def add(a, b):\n    return a + b"
            if "Fixture passage:" in text:
                value = "blue"
            outcome = re.search(r"\[fixture outcome: (correct|wrong|error)\]", text)
            if outcome and outcome[1] == "wrong":
                value = "def add(a, b):\n    return a - b" if "Fixture code:" in text else "red" if "Fixture passage:" in text else "-999"
            if outcome and outcome[1] == "error":
                if "Fixture code:" in text:
                    value = "def add(a, b):\n    raise ValueError('intentional fixture failure')"
                else:
                    # Structured-response workflows must reject malformed JSON;
                    # plain-text workflows receive an invalid final answer.
                    return {"role": "assistant", "content": "INTENTIONAL_INVALID_FIXTURE_RESPONSE"}
            result = json.dumps({f: value for f in fields}) if fields else value
        return {"role": "assistant", "content": result}
