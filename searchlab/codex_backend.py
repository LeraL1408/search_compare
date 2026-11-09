"""Fresh Codex CLI invocations, normalized to the shared model interface."""
from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import tempfile
from pathlib import Path

from .common import append


class CodexFailure(Exception):
    def __init__(self, message, usage=None):
        super().__init__(message)
        self.usage = usage


def codex_environment():
    # Retain saved ChatGPT login, never inherit API keys or endpoint overrides.
    allowed = {"PATH", "HOME", "CODEX_HOME", "TMPDIR", "LANG", "LC_ALL", "SSL_CERT_FILE",
               "SSL_CERT_DIR", "HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "NO_PROXY"}
    return {k: v for k, v in os.environ.items() if k in allowed}


def response_schema():
    return {"type": "object", "additionalProperties": False,
            "properties": {"content": {"type": "string"}, "tool_calls": {
                "type": "array", "items": {"type": "object", "additionalProperties": False,
                    "properties": {"name": {"type": "string"}, "arguments": {"type": "string"}},
                    "required": ["name", "arguments"]}}}, "required": ["content", "tool_calls"]}


def build_command(config, cwd, schema):
    command = [config.get("binary", "codex"), "exec", "--json", "--ephemeral", "--ignore-user-config",
               "--skip-git-repo-check", "--sandbox", "read-only", "--color", "never", "--cd", str(cwd),
               "--model", config["model"], "--output-schema", str(schema)]
    settings = {"forced_login_method": "chatgpt", "model_provider": "openai", "web_search": "disabled",
                "project_doc_max_bytes": 0, "history.persistence": "none", "mcp_servers": {},
                "model_reasoning_effort": config.get("reasoning_effort", "low")}
    for feature in ("shell_tool", "apps", "plugins", "multi_agent", "browser_use", "computer_use",
                    "in_app_browser", "view_image", "code_mode_host", "artifact", "shell_snapshot"):
        settings["features." + feature] = False
    for key, value in settings.items():
        # JSON strings/bools also have valid TOML scalar syntax; {} is a TOML table.
        command.extend(["-c", key + "=" + json.dumps(value)])
    command.append("-")
    return command


def parse_events(stdout, event_path):
    final, usage = None, None
    completed_turns = 0
    invalid = None
    for line in stdout.splitlines():
        if not line.strip():
            continue
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            invalid = "Codex emitted a non-JSON event"
            continue
        append(event_path, event)
        kind = event.get("type")
        if kind in ("item.started", "item.updated", "item.completed"):
            item = event.get("item", {})
            # CLI diagnostics are item events too, not tool invocations. A
            # successful completed turn can contain a startup warning.
            if item.get("type") not in ("agent_message", "reasoning", "error"):
                invalid = "Codex used a native tool outside the shared benchmark interface"
            if kind == "item.completed" and item.get("type") == "agent_message":
                final = item.get("text")
        if kind == "turn.completed":
            completed_turns += 1
            raw = event.get("usage") or {}
            if all(type(raw.get(k)) is int and raw[k] >= 0 for k in ("input_tokens", "output_tokens")):
                usage = {"prompt_tokens": raw["input_tokens"], "completion_tokens": raw["output_tokens"],
                         "cached_input_tokens": raw.get("cached_input_tokens", 0), "turns": completed_turns,
                         "reported": raw}
        if kind == "turn.failed":
            invalid = "Codex turn failed"
    if invalid:
        raise CodexFailure(invalid, usage)
    if not final or usage is None or completed_turns != 1:
        raise CodexFailure("Codex did not return exactly one completed turn with answer and token usage", usage)
    return final, usage


def normalize_response(content, tools):
    envelope = json.loads(content)
    if not isinstance(envelope.get("content"), str) or not isinstance(envelope.get("tool_calls"), list):
        raise ValueError("Invalid Codex response envelope")
    allowed = {t["function"]["name"] for t in tools or []}
    calls = []
    for i, call in enumerate(envelope["tool_calls"]):
        if call.get("name") not in allowed:
            raise ValueError("Codex requested an unavailable benchmark tool")
        arguments = call.get("arguments")
        if not isinstance(arguments, str) or not isinstance(json.loads(arguments), dict):
            raise ValueError("Tool arguments must encode a JSON object")
        calls.append({"id": f"codex-call-{i}", "type": "function", "function": {
            "name": call["name"], "arguments": arguments}})
    return {"role": "assistant", "content": envelope["content"], **({"tool_calls": calls} if calls else {})}


def invoke(config, messages, tools, fields, timeout, event_path, on_start=None):
    instruction = (
        "You are the model component of a controlled agent experiment. Respond only from the conversation below. "
        "Do not inspect files, run commands, browse, delegate, or invoke native Codex tools. "
        "Return the required envelope. Put your actual answer in content. "
        "If benchmark tools are needed, put their names and JSON-encoded argument objects in tool_calls, "
        "and leave content empty. These tool calls will be executed by the external benchmark, not by you. "
        "If no tool is needed, return tool_calls as an empty array. "
    )
    if fields:
        instruction += "The content string must itself encode a JSON object with these fields: " + json.dumps(fields) + ". "
    payload = instruction + "\nCONVERSATION AND AVAILABLE BENCHMARK TOOLS:\n" + json.dumps({"messages": messages, "tools": tools or []}, ensure_ascii=False)
    if len(payload.encode()) > config.get("max_prompt_bytes", 200_000):
        raise ValueError("Codex prompt exceeds configured byte limit")
    with tempfile.TemporaryDirectory(prefix="searchlab-codex-") as temporary:
        cwd = Path(temporary)
        schema = cwd / "response.schema.json"
        schema.write_text(json.dumps(response_schema()))
        command = build_command(config, cwd, schema)
        with tempfile.TemporaryFile() as error_stream:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                       stderr=error_stream, cwd=cwd, env=codex_environment(), start_new_session=True)
            if on_start:
                on_start()
            try:
                stdout, _ = process.communicate(payload.encode(), timeout=timeout)
            except BaseException:
                if process.poll() is None:
                    os.killpg(process.pid, signal.SIGKILL)
                stdout, _ = process.communicate()
                # A timed-out turn has no trustworthy complete usage; preserve
                # received events for diagnosis, never fabricate zero-token usage.
                try:
                    parse_events(stdout.decode(errors="replace"), event_path)
                except CodexFailure:
                    pass
                raise
            if len(stdout) > 4_000_000:
                raise CodexFailure("Codex event output exceeded the size limit")
            final, usage = None, None
            try:
                final, usage = parse_events(stdout.decode(errors="replace"), event_path)
                if process.returncode:
                    raise CodexFailure(f"Codex exited with status {process.returncode}", usage)
                message = normalize_response(final, tools)
            except Exception as exc:
                error_stream.seek(0)
                diagnostic = error_stream.read(4000).decode(errors="replace")
                diagnostic = re.sub(r"(?i)(bearer\s+|sk-)[A-Za-z0-9_.-]+", "[redacted]", diagnostic)
                append(event_path, {"type": "searchlab.codex_failure", "exit_code": process.returncode,
                                    "diagnostic": diagnostic})
                if isinstance(exc, CodexFailure):
                    raise
                raise CodexFailure(str(exc), usage) from exc
            return message, usage
