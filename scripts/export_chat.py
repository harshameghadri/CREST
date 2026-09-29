"""Export a Claude Code transcript (.jsonl) to a readable Markdown history (work.md).

    python scripts/export_chat.py ~/.claude/projects/<project>/<session>.jsonl work.md

Keeps user messages, assistant replies, compaction summaries and a one-line list of the
tool calls; drops tool output and reasoning; redacts tokens, e-mails, IPs and model names.
"""
import json, re, sys
from datetime import datetime

src, dst = sys.argv[1], sys.argv[2]

REDACT = [
    (re.compile(r"claude-(opus|sonnet|haiku|fable)[-\w.]*", re.I), "<model>"),
    (re.compile(r"\b(Claude\s+)?(Opus|Sonnet|Haiku|Fable)\s*\d+(\.\d+)?\b", re.I), "Claude"),
    (re.compile(r"\b(sk-ant-|sk-|ghp_|gho_|github_pat_|hf_)[A-Za-z0-9_\-]{16,}"), "<redacted-token>"),
    (re.compile(r"\b(fable|opus|sonnet|haiku)\b", re.I), "<model>"),
    (re.compile(r"[A-Za-z0-9._%+-]+@(?!anthropic\.com)[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "<email>"),
    (re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b(?!\s*(MB|GB|s\b))"), "<ip>"),
]
STRIP = [re.compile(p, re.S) for p in (
    r"<system-reminder>.*?</system-reminder>", r"<local-command-caveat>.*?</local-command-caveat>",
    r"<command-(name|message|args)>.*?</command-\1>", r"<local-command-stdout>.*?</local-command-stdout>",
    r"<task-notification>.*?</task-notification>")]
PASTE = re.compile(r"<pasted_content[^>]*>(.*?)</pasted_content[^>]*>", re.S)


def clean(t: str) -> str:
    for p in STRIP:
        t = p.sub("", t)
    t = PASTE.sub(lambda m: "\n```text\n" + m.group(1).strip() + "\n```\n", t)
    for p, r in REDACT:
        t = p.sub(r, t)
    t = re.sub(r"/tmp/claude-\d+/[^\s/]+/[0-9a-f-]{36}/(scratchpad/)?", "scratchpad/", t)
    t = re.sub(r"/root/\.claude/projects/[^\s]*?/", "<transcripts>/", t)
    return t.strip()


def ts(d):
    try:
        return datetime.fromisoformat(d["timestamp"].replace("Z", "+00:00")).strftime("%Y-%m-%d %H:%M UTC")
    except Exception:
        return ""


def tool_line(b):
    n, i = b.get("name", "?"), b.get("input", {}) or {}
    if n == "Bash":
        what = i.get("description") or i.get("command", "")[:80]
    elif n in ("Edit", "Write", "Read", "NotebookEdit"):
        what = str(i.get("file_path", "")).replace("/home/user/CREST/", "")
    elif n in ("Grep", "Glob"):
        what = i.get("pattern", "")
    elif n.startswith("Task"):
        what = i.get("subject") or i.get("status") or ""
    else:
        what = i.get("description") or i.get("title") or ""
    n = re.sub(r"^mcp__[^_]+(?:_[^_]+)*__", "", n) if n.startswith("mcp__") else n
    return f"{n}: {clean(str(what))[:120]}".rstrip(": ")


out = ["# CREST work history", "",
       "Export of the Claude Code session that built CREST 0.2.0 → 0.3.0 (Sept 2026): every user message,",
       "every assistant reply, and a one-line list of the actions (tool calls) taken in between.",
       "Tool outputs and internal reasoning are omitted. Context-compaction summaries are kept: they are",
       "dense recaps of everything before them and the fastest way to get the full picture.",
       "Tokens, e-mail addresses, IP addresses and model identifiers are redacted.", "",
       "**When in doubt about why something is the way it is, search this file.**", ""]
pending_tools = []


def flush_tools():
    global pending_tools
    if pending_tools:
        out.append("<details><summary>actions (" + str(len(pending_tools)) + ")</summary>\n")
        out.extend(f"- {l}" for l in pending_tools)
        out.append("\n</details>\n")
    pending_tools = []


n_user = n_asst = n_sum = 0
seen, latest = set(), ""
for line in open(src):
    d = json.loads(line)
    t = d.get("type")
    if t not in ("user", "assistant") or d.get("isSidechain"):
        continue
    # compaction replays earlier records (same uuid, or re-stamped copies with older timestamps)
    if d["uuid"] in seen or d.get("timestamp", "") < latest:
        continue
    seen.add(d["uuid"])
    latest = max(latest, d.get("timestamp", ""))
    msg = d.get("message") or {}
    content = msg.get("content")
    blocks = [{"type": "text", "text": content}] if isinstance(content, str) else (content or [])
    if t == "user":
        if d.get("isMeta"):
            continue
        texts = [b.get("text", "") for b in blocks if b.get("type") == "text"]
        if not texts:
            continue  # tool results
        txt = clean("\n".join(texts))
        if not txt or txt.startswith("[Request interrupted"):
            continue
        flush_tools()
        if d.get("isCompactSummary"):
            n_sum += 1
            txt = re.sub(r"^This session is being continued.*?\n\n", "", txt, flags=re.S)
            txt = re.sub(r"\n+If you need specific details from before compaction.*", "", txt, flags=re.S)
            out += ["", "---", "", f"## Context summary {n_sum} ({ts(d)})", "",
                    "<details><summary>Recap written when the conversation was compacted (click to expand)</summary>", "",
                    txt, "", "</details>", ""]
            continue
        n_user += 1
        out += ["", "---", "", f"### User ({ts(d)})", "", txt, ""]
    else:
        for b in blocks:
            if b.get("type") == "text" and b.get("text", "").strip():
                flush_tools()
                n_asst += 1
                out += [f"#### Claude ({ts(d)})", "", clean(b["text"]), ""]
            elif b.get("type") == "tool_use":
                pending_tools.append(tool_line(b))
flush_tools()
open(dst, "w").write("\n".join(out) + "\n")
print(f"{n_user} user messages, {n_asst} replies, {n_sum} summaries -> {dst}")
