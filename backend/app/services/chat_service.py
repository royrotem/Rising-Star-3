"""
AI Chat Service for UAIE

Provides conversational AI capabilities backed by Claude, with full
system data context awareness. Persists conversation history to disk.
This is an additive feature module — removing it does not affect core
analysis or query endpoints.
"""

import json
import math
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False


def _get_api_key() -> str:
    try:
        from ..api.app_settings import get_anthropic_api_key
        return get_anthropic_api_key()
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY", "")


def _sanitize(obj: Any) -> Any:
    """Recursively convert numpy/pandas types for JSON."""
    if isinstance(obj, dict):
        return {k: _sanitize(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_sanitize(i) for i in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (math.isnan(v) or math.isinf(v)) else v
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.ndarray):
        return _sanitize(obj.tolist())
    if isinstance(obj, float) and (math.isnan(obj) or math.isinf(obj)):
        return None
    return obj


# ── Conversation persistence ────────────────────────────────────────

class ConversationStore:
    """File-based conversation history storage, thread-safe."""

    def __init__(self, data_dir: str | None = None):
        if data_dir is None:
            data_dir = os.environ.get("DATA_DIR", "/app/data")
            if not os.path.exists("/app") and os.path.exists(os.path.dirname(__file__)):
                data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
        self.conv_dir = Path(data_dir) / "conversations"
        self.conv_dir.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()

    def _path(self, system_id: str) -> Path:
        return self.conv_dir / f"{system_id}.json"

    def get_messages(self, system_id: str, limit: int = 50) -> List[Dict]:
        with self._lock:
            p = self._path(system_id)
            if not p.exists():
                return []
            try:
                msgs = json.loads(p.read_text())
            except Exception:
                return []
            return msgs[-limit:]

    def add_message(self, system_id: str, role: str, content: str,
                    data: Dict | None = None) -> Dict:
        msg = {
            "id": f"msg-{datetime.utcnow().strftime('%Y%m%d%H%M%S%f')}",
            "role": role,
            "content": content,
            "timestamp": datetime.utcnow().isoformat(),
            "data": data,
        }
        with self._lock:
            p = self._path(system_id)
            msgs: list = []
            if p.exists():
                try:
                    msgs = json.loads(p.read_text())
                except Exception:
                    msgs = []
            msgs.append(msg)
            # Keep last 200 messages per system
            if len(msgs) > 200:
                msgs = msgs[-200:]
            p.write_text(json.dumps(msgs, indent=2, default=str))
        return msg

    def clear(self, system_id: str) -> None:
        with self._lock:
            p = self._path(system_id)
            if p.exists():
                p.unlink()


conversation_store = ConversationStore()


# ── Chat service ────────────────────────────────────────────────────

class ChatService:
    """
    AI-powered conversational interface for system data.

    Uses Claude with full context: system metadata, field statistics,
    recent analysis results, and conversation history.
    """

    MODEL = "claude-sonnet-4-20250514"

    def _get_client(self):
        api_key = _get_api_key()
        if HAS_ANTHROPIC and api_key:
            return anthropic.Anthropic(api_key=api_key)
        return None

    # ── Context builders ────────────────────────────────────────────

    def _build_system_context(self, system: Dict, records: List[Dict],
                              schema: list) -> str:
        """Build a concise context string about the system & its data."""
        lines = [
            f"System: {system.get('name', 'Unknown')}",
            f"Type: {system.get('system_type', 'unknown')}",
            f"Status: {system.get('status', 'active')}",
            f"Health Score: {system.get('health_score', 'N/A')}",
        ]

        meta = system.get("metadata", {})
        if meta.get("description"):
            lines.append(f"Description: {meta['description']}")

        if not records:
            lines.append("\nNo data ingested yet.")
            return "\n".join(lines)

        df = pd.DataFrame(records)
        lines.append(f"\nData: {len(df)} records, {len(df.columns)} fields")

        # Field stats
        lines.append("\nFields:")
        for col in df.columns[:30]:
            dtype = str(df[col].dtype)
            if df[col].dtype in ["int64", "float64"]:
                try:
                    lines.append(
                        f"  - {col} ({dtype}): "
                        f"min={df[col].min():.4g}, max={df[col].max():.4g}, "
                        f"mean={df[col].mean():.4g}, std={df[col].std():.4g}"
                    )
                except Exception:
                    lines.append(f"  - {col} ({dtype})")
            else:
                nunique = df[col].nunique()
                lines.append(f"  - {col} ({dtype}): {nunique} unique values")

        # Sample rows
        lines.append("\nSample rows (first 3):")
        for row in df.head(3).to_dict("records"):
            lines.append(f"  {json.dumps(_sanitize(row), default=str)[:400]}")

        # Schema field info
        if schema:
            confirmed = [f for f in schema if isinstance(f, dict) and f.get("confirmed")]
            if confirmed:
                lines.append("\nConfirmed field info:")
                for f in confirmed[:15]:
                    unit = f.get("physical_unit") or f.get("confirmed_unit", "")
                    meaning = f.get("inferred_meaning") or f.get("confirmed_meaning", "")
                    if unit or meaning:
                        lines.append(f"  - {f.get('name', '?')}: {meaning} ({unit})")

        return "\n".join(lines)

    def _system_prompt(self) -> str:
        return (
            "You are the UAIE Conversational Chief Engineer — an AI assistant "
            "embedded in the Universal Autonomous Insight Engine platform. "
            "You help engineers understand their system's telemetry data, "
            "investigate anomalies, and make data-driven decisions.\n\n"
            "Guidelines:\n"
            "- Be concise but thorough. Engineers appreciate precision.\n"
            "- Reference specific field names, values, and statistics from the data.\n"
            "- When explaining anomalies, provide possible root causes.\n"
            "- Suggest actionable next steps when appropriate.\n"
            "- If you don't have enough data to answer, say so clearly.\n"
            "- Format important values with **bold** and use bullet points for lists.\n"
            "- When asked about statistics, compute or reference the provided data.\n"
            "- You can reference field correlations and trends if available.\n"
        )

    # ── Main chat method ────────────────────────────────────────────

    async def chat(
        self,
        system_id: str,
        user_message: str,
        system: Dict,
        records: List[Dict],
        schema: list,
    ) -> Dict[str, Any]:
        """
        Process a user message and return an AI response.

        Falls back to a keyword-based approach if no API key is configured.
        """
        # Persist user message
        conversation_store.add_message(system_id, "user", user_message)

        # Build context
        system_context = self._build_system_context(system, records, schema)
        history = conversation_store.get_messages(system_id, limit=20)

        # Try AI-powered response
        client = self._get_client()
        if client:
            response = await self._ai_response(
                client, user_message, system_context, history
            )
        else:
            response = self._fallback_response(
                user_message, system, records
            )

        # Persist assistant response
        conversation_store.add_message(
            system_id, "assistant", response["content"], response.get("data")
        )

        return response

    async def _ai_response(
        self,
        client,
        user_message: str,
        system_context: str,
        history: List[Dict],
    ) -> Dict[str, Any]:
        """Generate response using Claude."""
        # Build message history for context (last 10 turns)
        messages = []
        for msg in history[-10:]:
            if msg["role"] in ("user", "assistant"):
                messages.append({
                    "role": msg["role"],
                    "content": msg["content"],
                })

        # Make sure the last message is the current user message
        if not messages or messages[-1]["content"] != user_message:
            messages.append({"role": "user", "content": user_message})

        # Ensure messages alternate and start with user
        clean_messages = []
        for msg in messages:
            if clean_messages and clean_messages[-1]["role"] == msg["role"]:
                # Merge consecutive same-role messages
                clean_messages[-1]["content"] += "\n" + msg["content"]
            else:
                clean_messages.append(msg)
        if clean_messages and clean_messages[0]["role"] != "user":
            clean_messages = clean_messages[1:]

        if not clean_messages:
            clean_messages = [{"role": "user", "content": user_message}]

        try:
            resp = client.messages.create(
                model=self.MODEL,
                max_tokens=2048,
                system=self._system_prompt() + f"\n\n--- SYSTEM DATA CONTEXT ---\n{system_context}",
                messages=clean_messages,
            )
            content = resp.content[0].text if resp.content else "I couldn't generate a response."

            return {
                "content": content,
                "ai_powered": True,
                "model": self.MODEL,
                "data": {"type": "ai_response"},
            }
        except Exception as e:
            print(f"[ChatService] Claude API error: {e}")
            return {
                "content": (
                    f"I encountered an error connecting to the AI service: {e}\n\n"
                    "Please check your API key in Settings."
                ),
                "ai_powered": False,
                "data": {"type": "error", "error": str(e)},
            }

    def _fallback_response(
        self,
        user_message: str,
        system: Dict,
        records: List[Dict],
    ) -> Dict[str, Any]:
        """Keyword-based fallback when no API key is available."""
        query = user_message.lower()

        if not records:
            return {
                "content": (
                    "No data has been ingested for this system yet. "
                    "Please upload telemetry data first, then I can help "
                    "you analyze it."
                ),
                "ai_powered": False,
                "data": {"type": "no_data"},
            }

        df = pd.DataFrame(records)
        numeric_cols = df.select_dtypes(include=["number"]).columns.tolist()

        if any(w in query for w in ["average", "mean"]):
            stats = {col: round(float(df[col].mean()), 4) for col in numeric_cols}
            lines = ["Here are the **average values** for numeric fields:\n"]
            for col, val in stats.items():
                lines.append(f"- **{col}**: {val}")
            return {
                "content": "\n".join(lines),
                "ai_powered": False,
                "data": {"type": "statistics", "statistics": _sanitize(stats)},
            }

        if any(w in query for w in ["max", "maximum", "highest", "peak"]):
            stats = {col: round(float(df[col].max()), 4) for col in numeric_cols}
            lines = ["Here are the **maximum values** for numeric fields:\n"]
            for col, val in stats.items():
                lines.append(f"- **{col}**: {val}")
            return {
                "content": "\n".join(lines),
                "ai_powered": False,
                "data": {"type": "statistics", "statistics": _sanitize(stats)},
            }

        if any(w in query for w in ["min", "minimum", "lowest"]):
            stats = {col: round(float(df[col].min()), 4) for col in numeric_cols}
            lines = ["Here are the **minimum values** for numeric fields:\n"]
            for col, val in stats.items():
                lines.append(f"- **{col}**: {val}")
            return {
                "content": "\n".join(lines),
                "ai_powered": False,
                "data": {"type": "statistics", "statistics": _sanitize(stats)},
            }

        if any(w in query for w in ["show", "find", "list", "get", "data"]):
            return {
                "content": (
                    f"The system has **{len(df)} records** across "
                    f"**{len(df.columns)} fields**: {', '.join(df.columns[:15])}.\n\n"
                    "You can ask me about averages, maximums, minimums, or "
                    "specific fields."
                ),
                "ai_powered": False,
                "data": {
                    "type": "data_overview",
                    "record_count": len(df),
                    "fields": list(df.columns),
                },
            }

        # Generic fallback
        return {
            "content": (
                f"System **{system.get('name', 'Unknown')}** has "
                f"**{len(df)} records** with **{len(df.columns)} fields**. "
                f"Health score: **{system.get('health_score', 'N/A')}**.\n\n"
                "Try asking:\n"
                "- \"What is the average temperature?\"\n"
                "- \"Show me the maximum values\"\n"
                "- \"What fields are available?\"\n\n"
                "*For AI-powered answers, configure your API key in Settings.*"
            ),
            "ai_powered": False,
            "data": {"type": "general"},
        }


chat_service = ChatService()
