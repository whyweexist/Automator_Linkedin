# app.py — LinkedIn Outreach Automation v2 (Single File)
# Run: streamlit run app.py
#
# pip install streamlit playwright openai anthropic google-generativeai groq sqlalchemy apscheduler

import streamlit as st
import asyncio
import threading
import json
import os
import re
import random
import time
import hashlib
import logging
import uuid
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple
from enum import Enum
from pathlib import Path
from contextlib import contextmanager

# AI providers (imported conditionally)
import openai

try:
    import anthropic

    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    # from google import genai

    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

try:
    import groq as groq_lib

    HAS_GROQ = True
except ImportError:
    HAS_GROQ = False

from sqlalchemy import (
    create_engine, Column, String, Integer, Float,
    DateTime, Text, Boolean, JSON as SA_JSON, inspect,
)
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.pool import StaticPool

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
LOG_FILE = "outreach.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE),
        logging.StreamHandler(),
    ],
)
logger = logging.getLogger("outreach")

# ─────────────────────────────────────────────
# DATABASE
# ─────────────────────────────────────────────
DB_PATH = "outreach.db"
Base = declarative_base()
engine = create_engine(
    f"sqlite:///{DB_PATH}",
    connect_args={"check_same_thread": False},
    poolclass=StaticPool,
    echo=False,
)
SessionLocal = sessionmaker(bind=engine)


class ProspectState(str, Enum):
    DISCOVERED = "discovered"
    PROFILE_SCRAPED = "profile_scraped"
    ANALYZED = "analyzed"
    CONNECTION_SENT = "connection_sent"
    CONNECTED = "connected"
    OPENER_SENT = "opener_sent"
    IN_CONVERSATION = "in_conversation"
    QUALIFIED = "qualified"
    NOT_INTERESTED = "not_interested"
    NO_RESPONSE = "no_response"
    NEEDS_HUMAN = "needs_human"
    ERROR = "error"


class ProspectModel(Base):
    __tablename__ = "prospects"
    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    profile_url = Column(String, unique=True, nullable=False)
    headline = Column(String, default="")
    location = Column(String, default="")
    state = Column(String, default=ProspectState.DISCOVERED.value)
    profile_data = Column(Text, default="{}")
    profile_insight = Column(Text, default="{}")
    conversation_history = Column(Text, default="[]")
    connection_note = Column(Text, default="")
    connection_sent_at = Column(DateTime, nullable=True)
    connection_accepted_at = Column(DateTime, nullable=True)
    last_message_at = Column(DateTime, nullable=True)
    last_message_from = Column(String, default="")
    follow_up_count = Column(Integer, default=0)
    score = Column(Float, default=0.0)
    tags = Column(Text, default="[]")
    source_query = Column(String, default="")
    notes = Column(Text, default="")
    error_message = Column(Text, default="")
    has_media_pending = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)


class ActivityLog(Base):
    __tablename__ = "activity_log"
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    action = Column(String, nullable=False)
    prospect_id = Column(String, default="")
    prospect_name = Column(String, default="")
    details = Column(Text, default="")
    status = Column(String, default="success")


class DailyStats(Base):
    __tablename__ = "daily_stats"
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, unique=True)
    connections_sent = Column(Integer, default=0)
    messages_sent = Column(Integer, default=0)
    profiles_scraped = Column(Integer, default=0)
    replies_received = Column(Integer, default=0)
    connections_accepted = Column(Integer, default=0)
    conversations_started = Column(Integer, default=0)
    prospects_discovered = Column(Integer, default=0)
    media_alerts = Column(Integer, default=0)
    new_dm_detected = Column(Integer, default=0)


class ConfigModel(Base):
    __tablename__ = "config"
    key = Column(String, primary_key=True)
    value = Column(Text)


class NotificationModel(Base):
    __tablename__ = "notifications"
    id = Column(String, primary_key=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    notification_type = Column(String, nullable=False)  # media_alert, new_unknown_dm, human_needed
    prospect_id = Column(String, default="")
    prospect_name = Column(String, default="")
    message = Column(Text, default="")
    media_type = Column(String, default="")  # image, video, file, audio
    is_read = Column(Boolean, default=False)
    is_resolved = Column(Boolean, default=False)
    details = Column(Text, default="{}")


class AIProviderConfig(Base):
    __tablename__ = "ai_providers"
    id = Column(String, primary_key=True)
    provider_name = Column(String, nullable=False)  # openai, anthropic, google, groq
    api_key = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    is_active = Column(Boolean, default=False)
    is_fallback = Column(Boolean, default=False)
    priority = Column(Integer, default=0)
    max_tokens = Column(Integer, default=200)
    temperature = Column(Float, default=0.85)
    label = Column(String, default="")  # user-friendly name
    created_at = Column(DateTime, default=datetime.utcnow)


Base.metadata.create_all(engine)


@contextmanager
def get_db():
    db = SessionLocal()
    try:
        yield db
        db.commit()
    except Exception:
        db.rollback()
        raise
    finally:
        db.close()


# ─────────────────────────────────────────────
# DATABASE HELPERS
# ─────────────────────────────────────────────
def ensure_column(engine, table, column, ddl):
    inspector = inspect(engine)
    cols = [c["name"] for c in inspector.get_columns(table)]
    if column not in cols:
        with engine.connect() as conn:
            conn.execute(text(ddl))
            conn.commit()

ensure_column(engine, "daily_stats", "media_alerts",
              "ALTER TABLE daily_stats ADD COLUMN media_alerts INTEGER DEFAULT 0")

ensure_column(engine, "daily_stats", "new_dm_detected",
              "ALTER TABLE daily_stats ADD COLUMN new_dm_detected INTEGER DEFAULT 0")
def _row_to_dict(r) -> dict:
    if r is None:
        return {}
    return {c.name: getattr(r, c.name) for c in r.__table__.columns}


class DB:
    @staticmethod
    def log_activity(action, prospect_id="", prospect_name="",
                     details="", status="success"):
        with get_db() as db:
            db.add(ActivityLog(
                action=action, prospect_id=prospect_id,
                prospect_name=prospect_name,
                details=str(details)[:2000], status=status,
                timestamp=datetime.utcnow(),
            ))

    @staticmethod
    def get_today_stats() -> dict:
        # today = datetime.utcnow().strftime("%Y-%m-%d")
        today = datetime.now(UTC).strftime("%Y-%m-%d")
        with get_db() as db:
            stats = db.query(DailyStats).filter_by(date=today).first()
            if not stats:
                stats = DailyStats(date=today)
                db.add(stats)
                db.commit()
            return _row_to_dict(stats)

    @staticmethod
    def increment_stat(field_name: str, amount: int = 1):
        today = datetime.utcnow().strftime("%Y-%m-%d")
        with get_db() as db:
            stats = db.query(DailyStats).filter_by(date=today).first()
            if not stats:
                stats = DailyStats(date=today)
                db.add(stats)
                db.flush()
            setattr(stats, field_name, getattr(stats, field_name, 0) + amount)

    @staticmethod
    def upsert_prospect(data: dict) -> str:
        pid = data.get("id", hashlib.md5(data["profile_url"].encode()).hexdigest()[:16])
        with get_db() as db:
            existing = db.query(ProspectModel).filter_by(id=pid).first()
            if existing:
                for k, v in data.items():
                    if k != "id":
                        setattr(existing, k, v)
                existing.updated_at = datetime.utcnow()
            else:
                data["id"] = pid
                data.setdefault("created_at", datetime.utcnow())
                data.setdefault("updated_at", datetime.utcnow())
                db.add(ProspectModel(**data))
        return pid

    @staticmethod
    def get_prospects_by_state(state: str) -> List[dict]:
        with get_db() as db:
            rows = db.query(ProspectModel).filter_by(state=state).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def get_prospect(pid: str) -> Optional[dict]:
        with get_db() as db:
            r = db.query(ProspectModel).filter_by(id=pid).first()
            return _row_to_dict(r) if r else None

    @staticmethod
    def get_prospect_by_name(name: str) -> Optional[dict]:
        with get_db() as db:
            r = db.query(ProspectModel).filter(
                ProspectModel.name.ilike(f"%{name}%")
            ).first()
            return _row_to_dict(r) if r else None

    @staticmethod
    def get_prospect_by_url(url: str) -> Optional[dict]:
        with get_db() as db:
            r = db.query(ProspectModel).filter(
                ProspectModel.profile_url.contains(url.split("?")[0])
            ).first()
            return _row_to_dict(r) if r else None

    @staticmethod
    def update_prospect(pid: str, data: dict):
        with get_db() as db:
            p = db.query(ProspectModel).filter_by(id=pid).first()
            if p:
                for k, v in data.items():
                    setattr(p, k, v)
                p.updated_at = datetime.utcnow()

    @staticmethod
    def get_all_prospects() -> List[dict]:
        with get_db() as db:
            rows = db.query(ProspectModel).order_by(
                ProspectModel.updated_at.desc()
            ).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def get_recent_activity(limit=50) -> List[dict]:
        with get_db() as db:
            rows = db.query(ActivityLog).order_by(
                ActivityLog.timestamp.desc()
            ).limit(limit).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def get_stats_history(days=30) -> List[dict]:
        with get_db() as db:
            # cutoff = (datetime.utcnow() - timedelta(days=days)).strftime("%Y-%m-%d")
            cutoff = (datetime.now(UTC) - timedelta(days=days)).strftime("%Y-%m-%d")
            rows = db.query(DailyStats).filter(
                DailyStats.date >= cutoff
            ).order_by(DailyStats.date).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def save_config(key: str, value: str):
        with get_db() as db:
            existing = db.query(ConfigModel).filter_by(key=key).first()
            if existing:
                existing.value = value
            else:
                db.add(ConfigModel(key=key, value=value))

    @staticmethod
    def get_config(key: str, default: str = "") -> str:
        with get_db() as db:
            row = db.query(ConfigModel).filter_by(key=key).first()
            return row.value if row else default

    # ── Notifications ──
    @staticmethod
    def add_notification(ntype: str, prospect_id="", prospect_name="",
                         message="", media_type="", details=""):
        with get_db() as db:
            db.add(NotificationModel(
                id=uuid.uuid4().hex[:16],
                notification_type=ntype,
                prospect_id=prospect_id,
                prospect_name=prospect_name,
                message=message,
                media_type=media_type,
                details=details,
                timestamp=datetime.utcnow(),
            ))

    @staticmethod
    def get_unread_notifications() -> List[dict]:
        with get_db() as db:
            rows = db.query(NotificationModel).filter_by(
                is_read=False
            ).order_by(NotificationModel.timestamp.desc()).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def get_all_notifications(limit=100) -> List[dict]:
        with get_db() as db:
            rows = db.query(NotificationModel).order_by(
                NotificationModel.timestamp.desc()
            ).limit(limit).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def mark_notification_read(nid: str):
        with get_db() as db:
            n = db.query(NotificationModel).filter_by(id=nid).first()
            if n:
                n.is_read = True

    @staticmethod
    def mark_notification_resolved(nid: str):
        with get_db() as db:
            n = db.query(NotificationModel).filter_by(id=nid).first()
            if n:
                n.is_read = True
                n.is_resolved = True

    @staticmethod
    def mark_all_notifications_read():
        with get_db() as db:
            db.query(NotificationModel).filter_by(
                is_read=False
            ).update({"is_read": True})

    # ── AI Providers ──
    @staticmethod
    def save_ai_provider(data: dict):
        pid = data.get("id", uuid.uuid4().hex[:16])
        with get_db() as db:
            existing = db.query(AIProviderConfig).filter_by(id=pid).first()
            if existing:
                for k, v in data.items():
                    if k != "id":
                        setattr(existing, k, v)
            else:
                data["id"] = pid
                db.add(AIProviderConfig(**data))
        return pid

    @staticmethod
    def get_ai_providers() -> List[dict]:
        with get_db() as db:
            rows = db.query(AIProviderConfig).order_by(
                AIProviderConfig.priority
            ).all()
            return [_row_to_dict(r) for r in rows]

    @staticmethod
    def get_active_ai_provider() -> Optional[dict]:
        with get_db() as db:
            r = db.query(AIProviderConfig).filter_by(
                is_active=True
            ).order_by(AIProviderConfig.priority).first()
            return _row_to_dict(r) if r else None

    @staticmethod
    def get_fallback_ai_provider() -> Optional[dict]:
        with get_db() as db:
            r = db.query(AIProviderConfig).filter_by(
                is_fallback=True
            ).order_by(AIProviderConfig.priority).first()
            return _row_to_dict(r) if r else None

    @staticmethod
    def set_active_provider(pid: str):
        with get_db() as db:
            db.query(AIProviderConfig).update({"is_active": False})
            p = db.query(AIProviderConfig).filter_by(id=pid).first()
            if p:
                p.is_active = True

    @staticmethod
    def delete_ai_provider(pid: str):
        with get_db() as db:
            db.query(AIProviderConfig).filter_by(id=pid).delete()


# ─────────────────────────────────────────────
# MULTI-PROVIDER AI ENGINE
# ─────────────────────────────────────────────
AVAILABLE_MODELS = {
    "openai": [
        "gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-4",
        "gpt-3.5-turbo", "o1-mini", "o1-preview",
    ],
    "anthropic": [
        "claude-sonnet-4-20250514", "claude-3-5-haiku-20241022",
        "claude-3-opus-20240229",
    ],
    "google": [
        "gemini-2.0-flash", "gemini-2.5-pro-preview-06-05",
        "gemini-2.5-flash-preview-05-20",
    ],
    "groq": [
        "llama-3.3-70b-versatile", "llama-3.1-8b-instant",
        "mixtral-8x7b-32768", "gemma2-9b-it",
    ],
}


class MultiModelAI:
    """AI engine supporting multiple providers with automatic fallback."""

    def __init__(self):
        self._clients = {}

    def _get_client(self, provider: str, api_key: str):
        cache_key = f"{provider}:{hashlib.md5(api_key.encode()).hexdigest()[:8]}"
        if cache_key not in self._clients:
            if provider == "openai":
                self._clients[cache_key] = openai.OpenAI(api_key=api_key)
            elif provider == "anthropic" and HAS_ANTHROPIC:
                self._clients[cache_key] = anthropic.Anthropic(api_key=api_key)
            elif provider == "google" and HAS_GOOGLE:
                genai.configure(api_key=api_key)
                self._clients[cache_key] = genai
            elif provider == "groq" and HAS_GROQ:
                self._clients[cache_key] = groq_lib.Groq(api_key=api_key)
            else:
                raise ValueError(f"Provider {provider} not available (missing library?)")
        return self._clients[cache_key]

    def chat(
        self,
        messages: List[dict],
        provider_config: dict = None,
        json_mode: bool = False,
        max_tokens: int = None,
        temperature: float = None,
    ) -> str:
        """Send chat completion with automatic fallback."""
        if not provider_config:
            provider_config = DB.get_active_ai_provider()
        if not provider_config:
            raise ValueError("No active AI provider configured!")

        provider = provider_config["provider_name"]
        api_key = provider_config["api_key"]
        model = provider_config["model_name"]
        mt = max_tokens or provider_config.get("max_tokens", 200)
        temp = temperature or provider_config.get("temperature", 0.85)

        try:
            return self._call_provider(
                provider, api_key, model, messages, json_mode, mt, temp
            )
        except Exception as e:
            logger.warning(f"Primary AI ({provider}/{model}) failed: {e}")
            # Try fallback
            fallback = DB.get_fallback_ai_provider()
            if fallback and fallback["id"] != provider_config.get("id"):
                logger.info(f"Trying fallback: {fallback['provider_name']}/{fallback['model_name']}")
                try:
                    return self._call_provider(
                        fallback["provider_name"], fallback["api_key"],
                        fallback["model_name"], messages, json_mode, mt, temp
                    )
                except Exception as e2:
                    logger.error(f"Fallback also failed: {e2}")
                    raise e2
            raise e

    def _call_provider(
        self, provider, api_key, model, messages, json_mode, max_tokens, temperature
    ) -> str:
        client = self._get_client(provider, api_key)

        if provider == "openai":
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()

        elif provider == "anthropic":
            system_msg = ""
            user_msgs = []
            for m in messages:
                if m["role"] == "system":
                    system_msg += m["content"] + "\n"
                else:
                    user_msgs.append(m)
            if not user_msgs:
                user_msgs = [{"role": "user", "content": "Hello"}]
            kwargs = {
                "model": model,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "messages": user_msgs,
            }
            if system_msg:
                kwargs["system"] = system_msg.strip()
            resp = client.messages.create(**kwargs)
            text = resp.content[0].text.strip()
            if json_mode:
                # Extract JSON from response
                match = re.search(r'\{.*\}', text, re.DOTALL)
                if match:
                    text = match.group(0)
            return text

        elif provider == "google":
            gmodel = client.GenerativeModel(model)
            combined = ""
            for m in messages:
                role_label = "System" if m["role"] == "system" else (
                    "User" if m["role"] == "user" else "Assistant"
                )
                combined += f"{role_label}: {m['content']}\n\n"
            gen_config = {
                "max_output_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                gen_config["response_mime_type"] = "application/json"
            resp = gmodel.generate_content(
                combined,
                generation_config=gen_config
            )
            return resp.text.strip()

        elif provider == "groq":
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "temperature": temperature,
            }
            if json_mode:
                kwargs["response_format"] = {"type": "json_object"}
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content.strip()

        raise ValueError(f"Unknown provider: {provider}")

    def test_connection(self, provider_config: dict) -> Tuple[bool, str]:
        """Test if a provider config works."""
        try:
            result = self._call_provider(
                provider_config["provider_name"],
                provider_config["api_key"],
                provider_config["model_name"],
                [{"role": "user", "content": "Say 'hello' in one word."}],
                False, 10, 0.5,
            )
            return True, f"Success: {result}"
        except Exception as e:
            return False, f"Failed: {str(e)}"


# ─────────────────────────────────────────────
# AI CONVERSATION ENGINE (uses MultiModelAI)
# ─────────────────────────────────────────────
class AIEngine:
    def __init__(self, multi_ai: MultiModelAI):
        self.ai = multi_ai

    def analyze_profile(self, profile_data: dict) -> dict:
        prompt = f"""Analyze this LinkedIn profile for outreach purposes.

PROFILE:
{json.dumps(profile_data, indent=2, default=str)}

Return JSON:
{{
    "current_role": "",
    "company_name": "",
    "company_stage": "idea|pre-seed|seed|series-a|growth|established|unknown",
    "what_they_are_building": "",
    "tech_stack_signals": [],
    "pain_points": ["likely challenge 1", "likely challenge 2"],
    "interests": [],
    "talking_points": ["specific thing from profile worth mentioning"],
    "tone_preference": "casual|professional|technical",
    "connection_angle": "best natural reason to connect",
    "conversation_hooks": [
        "specific question about their work",
        "question about a challenge they likely face"
    ],
    "relevance_score": 0-100
}}

Be SPECIFIC referencing actual profile details. Never be generic."""

        result = self.ai.chat(
            messages=[
                {"role": "system", "content": "You analyze LinkedIn profiles for genuine outreach. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            max_tokens=500,
            temperature=0.7,
        )
        return json.loads(result)

    def generate_connection_note(self, insight: dict, your_context: dict) -> str:
        prompt = f"""Write a LinkedIn connection request note (UNDER 280 chars).

THEM:
- Role: {insight.get('current_role','')}
- Building: {insight.get('what_they_are_building','')}
- Angle: {insight.get('connection_angle','')}
- Talking point: {(insight.get('talking_points') or ['N/A'])[0]}

YOU: {your_context.get('name','')}, {your_context.get('role','')}
You do: {your_context.get('what_you_do','')}

RULES:
- UNDER 280 characters
- Genuine, not templated
- Reference something specific about them
- No sales pitch, no flattery
- Casual but respectful
- Clear reason to connect

Output ONLY the note text."""

        note = self.ai.chat(
            messages=[
                {"role": "system", "content": "Write a short LinkedIn connection note. Output ONLY the note."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=100,
            temperature=0.9,
        )
        return note.strip().strip('"\'')[:295]

    def generate_opener(self, insight: dict, your_context: dict) -> str:
        system = self._build_system_prompt(insight, your_context)
        prompt = f"""They just accepted your connection request. Write your opening DM.

Rules:
- Brief thanks for connecting (not gushing)
- Reference something specific about what they're building
- Ask ONE genuine question
- 2-3 sentences max
- Make them want to reply

Best hook: {(insight.get('conversation_hooks') or ['ask about their project'])[0]}
Talking point: {(insight.get('talking_points') or [''])[0]}

Output ONLY the message."""

        result = self.ai.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": prompt},
            ],
            max_tokens=150,
            temperature=0.85,
        )
        return result.strip().strip('"')

    def generate_reply(self, insight, your_context, conversation_history, their_message):
        system = self._build_system_prompt(insight, your_context)
        messages = [{"role": "system", "content": system}]

        for msg in conversation_history[-8:]:
            role = "assistant" if msg.get("sender") == "me" else "user"
            messages.append({"role": role, "content": msg["text"]})

        messages.append({
            "role": "user",
            "content": (
                f'Their latest message: "{their_message}"\n\n'
                "Reply naturally as a DM. Keep it short (2-4 sentences). "
                "Ask at most ONE question. Be genuinely curious. "
                "Output ONLY your reply message."
            ),
        })

        result = self.ai.chat(messages=messages, max_tokens=200, temperature=0.85)
        return result.strip().strip('"')

    def classify_intent(self, message: str, context: list) -> dict:
        prompt = f"""Classify this LinkedIn DM reply.

Message: "{message}"

Recent context:
{json.dumps(context[-4:], indent=2, default=str)}

Return JSON:
{{
    "sentiment": "positive|neutral|negative|not_interested",
    "engagement_level": "high|medium|low|disengaged",
    "asks_question": true,
    "open_to_continue": true,
    "suggested_action": "reply|share_resource|propose_call|back_off|wait",
    "urgency": "respond_quickly|normal|can_wait",
    "key_topics": []
}}"""

        result = self.ai.chat(
            messages=[
                {"role": "system", "content": "Classify DM intent. Return JSON only."},
                {"role": "user", "content": prompt},
            ],
            json_mode=True,
            max_tokens=150,
            temperature=0.3,
        )
        return json.loads(result)

    def generate_follow_up(self, insight, your_context, history, attempt):
        styles = {
            1: "Send a casual value-add follow-up. Share something useful related to their work. Very brief.",
            2: "Send a final light follow-up. A friendly 'no worries if busy' nudge. Super short.",
        }
        style = styles.get(attempt, styles[2])
        system = self._build_system_prompt(insight, your_context)

        messages = [{"role": "system", "content": system}]
        for msg in history[-4:]:
            role = "assistant" if msg.get("sender") == "me" else "user"
            messages.append({"role": role, "content": msg["text"]})
        messages.append({
            "role": "user",
            "content": f"[They haven't replied. Follow-up #{attempt}] {style}\n\nOutput ONLY the message.",
        })

        result = self.ai.chat(messages=messages, max_tokens=120, temperature=0.9)
        return result.strip().strip('"')

    def generate_new_dm_reply(self, sender_name: str, their_message: str,
                              sender_profile: dict, your_context: dict) -> str:
        """Generate a reply for a completely new DM from someone not in pipeline."""
        system = f"""You are {your_context.get('name','')}, {your_context.get('role','')}.
Someone you don't know well just sent you a LinkedIn DM. Reply naturally.

Your background: {json.dumps(your_context, indent=2, default=str)}

About them (if known):
{json.dumps(sender_profile, indent=2, default=str)}

RULES:
1. Be friendly, genuine, and curious
2. Keep it short (2-3 sentences)
3. If they're pitching you, be polite but direct
4. If they're asking for help, be helpful but brief
5. If it's a genuine conversation starter, engage naturally
6. Sound like a real person, not a bot
7. Match their tone"""

        result = self.ai.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": f'{sender_name} says: "{their_message}"\n\nReply naturally. Output ONLY the message.'},
            ],
            max_tokens=150,
            temperature=0.85,
        )
        return result.strip().strip('"')

    def _build_system_prompt(self, insight: dict, your_context: dict) -> str:
        return f"""You are a real human having a LinkedIn DM conversation.

YOU ARE: {your_context.get('name','')}, {your_context.get('role','')}
YOUR BACKGROUND: {json.dumps(your_context, indent=2, default=str)}

TALKING TO:
- Role: {insight.get('current_role','')}
- Company: {insight.get('company_name','')}
- Building: {insight.get('what_they_are_building','')}
- Pain points: {', '.join(insight.get('pain_points',[]))}
- Tone: {insight.get('tone_preference','casual')}

RULES:
1. Sound like a REAL person, never a bot or salesperson
2. Short messages (2-4 sentences usually)
3. Sometimes lowercase, imperfect grammar is OK
4. Genuine curiosity about their work
5. Share your experiences naturally when relevant
6. ONE question per message max
7. Reference specifics from their profile naturally
8. Have opinions — don't just agree
9. Use filler words occasionally ("honestly", "tbh", "actually")
10. Match their tone and energy
11. Emoji sparingly (max 1, not every msg)
12. Vary message length
13. Never reveal AI assistance
14. Build toward understanding needs — never pitch directly"""


# ─────────────────────────────────────────────
# MEDIA DETECTOR
# ─────────────────────────────────────────────
class MediaDetector:
    """Detects images, videos, files, and other media in LinkedIn DMs."""

    IMAGE_INDICATORS = [
        ".jpg", ".jpeg", ".png", ".gif", ".webp", ".svg", ".bmp",
        "feed-shared-image", "msg-s-event-listitem__image",
        "ivm-image-view-model", "pv-profile-image",
        "media-container", "img-wrapper",
    ]

    VIDEO_INDICATORS = [
        ".mp4", ".mov", ".avi", ".webm",
        "feed-shared-video", "video-player",
        "msg-media-container--video", "li-video",
        "video-embed", "youtube.com", "vimeo.com",
        "loom.com", "wistia.com",
    ]

    FILE_INDICATORS = [
        ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
        ".zip", ".csv", ".txt",
        "msg-s-event-listitem__attachment",
        "file-attachment", "document-attachment",
    ]

    AUDIO_INDICATORS = [
        ".mp3", ".wav", ".ogg", ".m4a",
        "voice-message", "audio-message",
        "msg-voice-", "voice-note",
    ]

    @classmethod
    async def detect_media_in_message(cls, page, message_element) -> dict:
        """Check a message element for media content."""
        result = {
            "has_media": False,
            "media_types": [],
            "details": [],
        }

        try:
            outer_html = await message_element.evaluate("el => el.outerHTML")
            html_lower = outer_html.lower()

            # Check images
            img_els = await message_element.query_selector_all("img:not(.presence-entity__image)")
            if img_els:
                for img in img_els:
                    src = await img.get_attribute("src") or ""
                    alt = await img.get_attribute("alt") or ""
                    if src and "emoji" not in src.lower() and "icon" not in src.lower():
                        result["has_media"] = True
                        if "image" not in result["media_types"]:
                            result["media_types"].append("image")
                        result["details"].append(f"Image: {alt or src[:80]}")

            for indicator in cls.IMAGE_INDICATORS:
                if indicator in html_lower and "image" not in result["media_types"]:
                    result["has_media"] = True
                    result["media_types"].append("image")
                    result["details"].append(f"Image detected (indicator: {indicator})")
                    break

            # Check videos
            video_els = await message_element.query_selector_all("video, iframe")
            if video_els:
                result["has_media"] = True
                if "video" not in result["media_types"]:
                    result["media_types"].append("video")
                result["details"].append("Embedded video/iframe found")

            for indicator in cls.VIDEO_INDICATORS:
                if indicator in html_lower and "video" not in result["media_types"]:
                    result["has_media"] = True
                    result["media_types"].append("video")
                    result["details"].append(f"Video detected (indicator: {indicator})")
                    break

            # Check files/attachments
            attachment_els = await message_element.query_selector_all(
                '[class*="attachment"], [class*="document"], a[href*=".pdf"], a[href*=".doc"]'
            )
            if attachment_els:
                result["has_media"] = True
                if "file" not in result["media_types"]:
                    result["media_types"].append("file")
                for att in attachment_els:
                    att_text = await att.inner_text()
                    result["details"].append(f"File: {att_text[:60]}")

            for indicator in cls.FILE_INDICATORS:
                if indicator in html_lower and "file" not in result["media_types"]:
                    result["has_media"] = True
                    result["media_types"].append("file")
                    result["details"].append(f"File detected (indicator: {indicator})")
                    break

            # Check audio/voice
            for indicator in cls.AUDIO_INDICATORS:
                if indicator in html_lower and "audio" not in result["media_types"]:
                    result["has_media"] = True
                    result["media_types"].append("audio")
                    result["details"].append(f"Audio detected (indicator: {indicator})")
                    break

            # GIF detection
            gif_els = await message_element.query_selector_all(
                '[class*="gif"], [data-test*="gif"]'
            )
            if gif_els or ".gif" in html_lower:
                result["has_media"] = True
                if "gif" not in result["media_types"]:
                    result["media_types"].append("gif")
                result["details"].append("GIF detected")

        except Exception as e:
            logger.debug(f"Media detection error: {e}")

        return result

    @classmethod
    def detect_media_from_text(cls, text: str) -> dict:
        """Fallback text-based media detection when DOM inspection isn't possible."""
        result = {"has_media": False, "media_types": [], "details": []}
        text_lower = text.lower()

        # Check for image-related text
        image_phrases = [
            "sent a photo", "sent an image", "shared a photo",
            "shared an image", "[image]", "[photo]", "📸", "🖼️",
            "check this screenshot", "see attached image",
            "here's a screenshot",
        ]
        for phrase in image_phrases:
            if phrase in text_lower:
                result["has_media"] = True
                result["media_types"].append("image")
                result["details"].append(f"Text indicates image: '{phrase}'")
                break

        # Check for video
        video_phrases = [
            "sent a video", "shared a video", "[video]",
            "check this video", "watch this", "🎥", "📹",
            "here's a video", "youtube.com", "youtu.be",
            "loom.com", "vimeo.com",
        ]
        for phrase in video_phrases:
            if phrase in text_lower:
                result["has_media"] = True
                result["media_types"].append("video")
                result["details"].append(f"Text indicates video: '{phrase}'")
                break

        # Check for files
        file_phrases = [
            "sent a file", "shared a document", "[file]",
            "[attachment]", "attached", "📎", "📄",
            ".pdf", ".doc", ".xlsx", ".pptx",
        ]
        for phrase in file_phrases:
            if phrase in text_lower:
                result["has_media"] = True
                result["media_types"].append("file")
                result["details"].append(f"Text indicates file: '{phrase}'")
                break

        # Audio
        audio_phrases = [
            "sent a voice message", "voice note", "[audio]",
            "🎤", "🔊", "voice message",
        ]
        for phrase in audio_phrases:
            if phrase in text_lower:
                result["has_media"] = True
                result["media_types"].append("audio")
                result["details"].append(f"Text indicates audio: '{phrase}'")
                break

        return result


# ─────────────────────────────────────────────
# HUMAN BEHAVIOR SIMULATOR
# ─────────────────────────────────────────────
class HumanBehavior:
    @staticmethod
    async def random_delay(mn=1.5, mx=4.5):
        await asyncio.sleep(random.uniform(mn, mx))

    @staticmethod
    async def human_type(page, selector, text):
        await page.click(selector)
        await asyncio.sleep(random.uniform(0.3, 0.8))
        for char in text:
            await page.keyboard.type(char)
            base = random.uniform(0.04, 0.12)
            if random.random() < 0.05:
                await asyncio.sleep(random.uniform(0.3, 1.2))
            elif char in ".!?,":
                await asyncio.sleep(random.uniform(0.15, 0.4))
            else:
                await asyncio.sleep(base)

    @staticmethod
    async def human_scroll(page):
        for _ in range(random.randint(2, 5)):
            await page.mouse.wheel(0, random.randint(150, 500))
            await asyncio.sleep(random.uniform(0.2, 0.6))
        if random.random() < 0.3:
            await page.mouse.wheel(0, -random.randint(50, 150))

    @staticmethod
    def is_active_hours(config: dict) -> bool:
        now = datetime.now()
        start = config.get("active_start_hour", 9)
        end = config.get("active_end_hour", 19)
        return start <= now.hour < end

    @staticmethod
    def get_reply_delay(intent: dict) -> float:
        if intent.get("urgency") == "respond_quickly":
            return random.uniform(60, 300)
        elif intent.get("asks_question"):
            return random.uniform(120, 600)
        return random.uniform(300, 1800)


# ─────────────────────────────────────────────
# LINKEDIN BROWSER CLIENT
# ─────────────────────────────────────────────
class LinkedInBrowser:
    def __init__(self):
        self.browser = None
        self.page = None
        self.context = None
        self.human = HumanBehavior()
        self._running = False
        self._playwright = None

    async def launch(self):
        from playwright.async_api import async_playwright
        self._playwright = await async_playwright().start()
        self.browser = await self._playwright.chromium.launch(
            headless=True,
            args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
        )
        self.context = await self.browser.new_context(
            viewport={"width": 1920, "height": 1080},
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/122.0.0.0 Safari/537.36"
            ),
            locale="en-US",
            timezone_id="America/New_York",
        )
        await self.context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            window.chrome = {runtime: {}};
        """)
        self.page = await self.context.new_page()
        self._running = True

    async def close(self):
        self._running = False
        if self.browser:
            await self.browser.close()
        if self._playwright:
            await self._playwright.stop()

    async def login(self, email: str, password: str) -> bool:
        cookie_file = Path("linkedin_cookies.json")
        if cookie_file.exists():
            cookies = json.loads(cookie_file.read_text())
            await self.context.add_cookies(cookies)
            await self.page.goto("https://www.linkedin.com/feed/")
            await asyncio.sleep(4)
            if "/feed" in self.page.url:
                logger.info("Logged in via cookies")
                return True

        await self.page.goto("https://www.linkedin.com/login")
        await self.human.random_delay(2, 4)
        await self.human.human_type(self.page, "#username", email)
        await self.human.random_delay(0.5, 1.5)
        await self.human.human_type(self.page, "#password", password)
        await self.human.random_delay(0.5, 1.0)
        await self.page.click('button[data-litms-control-urn="login-submit"]')
        await self.page.wait_for_load_state("networkidle", timeout=30000)
        await self.human.random_delay(3, 6)

        if "/feed" in self.page.url or "/mynetwork" in self.page.url:
            cookies = await self.context.cookies()
            cookie_file.write_text(json.dumps(cookies))
            logger.info("LinkedIn login successful")
            return True
        logger.error(f"Login may have failed — URL: {self.page.url}")
        return False

    async def search_people(self, query: str, max_results: int = 10) -> List[dict]:
        url = (
            f"https://www.linkedin.com/search/results/people/"
            f"?keywords={query}&origin=GLOBAL_SEARCH_HEADER"
        )
        await self.page.goto(url)
        await self.human.random_delay(3, 6)
        await self.human.human_scroll(self.page)
        await self.human.random_delay(1, 3)

        results = []
        cards = await self.page.query_selector_all(".reusable-search__result-container")
        for card in cards[:max_results]:
            try:
                name_el = await card.query_selector(
                    "span.entity-result__title-text a span[dir='ltr'] > span"
                )
                headline_el = await card.query_selector(".entity-result__primary-subtitle")
                location_el = await card.query_selector(".entity-result__secondary-subtitle")
                link_el = await card.query_selector(".entity-result__title-text a")
                if not name_el or not link_el:
                    continue
                name = (await name_el.inner_text()).strip()
                headline = (await headline_el.inner_text()).strip() if headline_el else ""
                location = (await location_el.inner_text()).strip() if location_el else ""
                href = await link_el.get_attribute("href")
                profile_url = href.split("?")[0] if href else ""
                if not profile_url:
                    continue
                results.append({
                    "name": name, "headline": headline,
                    "location": location, "profile_url": profile_url,
                })
            except Exception:
                continue
            await self.human.random_delay(0.3, 0.8)
        return results

    async def scrape_profile(self, profile_url: str) -> dict:
        await self.page.goto(profile_url)
        await self.human.random_delay(3, 5)
        for _ in range(3):
            await self.human.human_scroll(self.page)
            await self.human.random_delay(1, 2)

        async def _txt(sel):
            el = await self.page.query_selector(sel)
            return (await el.inner_text()).strip() if el else ""

        data = {
            "url": profile_url,
            "name": await _txt(".text-heading-xlarge"),
            "headline": await _txt(".text-body-medium.break-words"),
            "location": await _txt(".pv-text-details__left-panel .text-body-small"),
            "about": await _txt("#about ~ .display-flex .pv-shared-text-with-see-more span"),
        }

        experience = []
        exp_items = await self.page.query_selector_all(
            "#experience ~ .pvs-list__outer-container .pvs-entity--padded"
        )
        for item in exp_items[:5]:
            async def _itxt(sel):
                el = await item.query_selector(sel)
                return (await el.inner_text()).strip() if el else ""
            experience.append({
                "title": await _itxt(".t-bold .visually-hidden"),
                "company": await _itxt(".t-14.t-normal .visually-hidden"),
                "duration": await _itxt(".t-14.t-normal.t-black--light .visually-hidden"),
            })
        data["experience"] = experience

        skills = []
        skill_els = await self.page.query_selector_all(
            "#skills ~ .pvs-list__outer-container .t-bold .visually-hidden"
        )
        for s in skill_els[:10]:
            t = await s.inner_text()
            if t:
                skills.append(t.strip())
        data["skills"] = skills

        posts = []
        try:
            await self.page.goto(f"{profile_url}/recent-activity/all/")
            await self.human.random_delay(3, 5)
            await self.human.human_scroll(self.page)
            post_els = await self.page.query_selector_all(".feed-shared-update-v2")
            for p in post_els[:3]:
                content_el = await p.query_selector(".feed-shared-text .break-words")
                if content_el:
                    posts.append((await content_el.inner_text()).strip()[:500])
        except Exception:
            pass
        data["recent_posts"] = posts
        return data

    async def send_connection_request(self, profile_url: str, note: str = "") -> bool:
        await self.page.goto(profile_url)
        await self.human.random_delay(2, 4)

        connect_btn = await self.page.query_selector('button:has-text("Connect")')
        if not connect_btn:
            more_btn = await self.page.query_selector('button:has-text("More")')
            if more_btn:
                await more_btn.click()
                await self.human.random_delay(1, 2)
                connect_btn = await self.page.query_selector('div[role="menuitem"]:has-text("Connect")')

        if not connect_btn:
            return False

        await connect_btn.click()
        await self.human.random_delay(1, 2)

        if note:
            add_note_btn = await self.page.query_selector('button:has-text("Add a note")')
            if add_note_btn:
                await add_note_btn.click()
                await self.human.random_delay(0.5, 1)
                textarea = await self.page.query_selector('textarea[name="message"]')
                if textarea:
                    await self.human.human_type(self.page, 'textarea[name="message"]', note[:295])
                    await self.human.random_delay(0.5, 1)

        send_btn = await self.page.query_selector('button:has-text("Send")')
        if send_btn:
            await send_btn.click()
            await self.human.random_delay(2, 4)
            return True
        return False

    async def check_connection_accepted(self, profile_url: str) -> bool:
        await self.page.goto(profile_url)
        await self.human.random_delay(2, 4)
        msg_btn = await self.page.query_selector('button:has-text("Message")')
        pending = await self.page.query_selector('button:has-text("Pending")')
        return msg_btn is not None and pending is None

    async def send_message(self, profile_url: str, message: str) -> bool:
        await self.page.goto(profile_url)
        await self.human.random_delay(2, 4)

        msg_btn = await self.page.query_selector('button:has-text("Message")')
        if not msg_btn:
            return False

        await msg_btn.click()
        await self.human.random_delay(1, 3)

        msg_box = await self.page.query_selector(
            'div.msg-form__contenteditable[contenteditable="true"]'
        )
        if not msg_box:
            msg_box = await self.page.query_selector('div[role="textbox"]')
        if not msg_box:
            return False

        await msg_box.click()
        await self.human.random_delay(0.3, 0.8)

        for char in message:
            await self.page.keyboard.type(char)
            await asyncio.sleep(random.uniform(0.03, 0.1))
            if random.random() < 0.04:
                await asyncio.sleep(random.uniform(0.3, 1.0))

        await self.human.random_delay(0.5, 1.5)

        send_btn = await self.page.query_selector('button.msg-form__send-button')
        if not send_btn:
            send_btn = await self.page.query_selector('button:has-text("Send")')
        if send_btn:
            await send_btn.click()
            await self.human.random_delay(2, 4)
            return True
        return False

    async def get_all_unread_threads(self) -> List[dict]:
        """Check messaging inbox for ALL unread messages — new DMs included."""
        await self.page.goto("https://www.linkedin.com/messaging/")
        await self.human.random_delay(3, 5)

        unread = []
        # Scroll to load more threads
        for _ in range(2):
            await self.human.human_scroll(self.page)
            await self.human.random_delay(1, 2)

        threads = await self.page.query_selector_all(
            ".msg-conversation-listitem--unread"
        )

        for thread in threads[:15]:
            try:
                name_el = await thread.query_selector(
                    ".msg-conversation-listitem__participant-names"
                )
                preview_el = await thread.query_selector(
                    ".msg-conversation-card__message-snippet-body"
                )
                link_el = await thread.query_selector("a")
                time_el = await thread.query_selector("time")

                if not name_el:
                    continue

                name = (await name_el.inner_text()).strip()
                preview = ""
                if preview_el:
                    preview = (await preview_el.inner_text()).strip()
                href = ""
                if link_el:
                    href = await link_el.get_attribute("href") or ""
                timestamp = ""
                if time_el:
                    timestamp = await time_el.get_attribute("datetime") or ""

                # Check for media indicators in preview
                media_in_preview = MediaDetector.detect_media_from_text(preview)

                unread.append({
                    "name": name,
                    "preview": preview,
                    "thread_url": href,
                    "timestamp": timestamp,
                    "media_detected_preview": media_in_preview,
                })
            except Exception:
                continue
        return unread

    async def read_conversation_with_media_check(self, thread_url: str) -> dict:
        """Read messages from a thread AND check for media content."""
        full_url = (
            f"https://www.linkedin.com{thread_url}"
            if thread_url.startswith("/")
            else thread_url
        )
        await self.page.goto(full_url)
        await self.human.random_delay(2, 4)

        messages = []
        media_messages = []

        msg_els = await self.page.query_selector_all(".msg-s-event-listitem")

        for m in msg_els[-15:]:
            try:
                sender_el = await m.query_selector(".msg-s-message-group__name")
                body_el = await m.query_selector(".msg-s-event-listitem__body")
                time_el = await m.query_selector("time")

                sender = (await sender_el.inner_text()).strip() if sender_el else ""
                body = (await body_el.inner_text()).strip() if body_el else ""
                ts = await time_el.get_attribute("datetime") if time_el else ""

                # Check for media in this message element
                media_result = await MediaDetector.detect_media_in_message(self.page, m)

                # Also check text-based indicators
                text_media = MediaDetector.detect_media_from_text(body)
                if text_media["has_media"]:
                    media_result["has_media"] = True
                    media_result["media_types"].extend(text_media["media_types"])
                    media_result["details"].extend(text_media["details"])
                    media_result["media_types"] = list(set(media_result["media_types"]))

                msg_data = {
                    "sender": sender,
                    "text": body,
                    "timestamp": ts,
                    "has_media": media_result["has_media"],
                    "media_types": media_result["media_types"],
                    "media_details": media_result["details"],
                }
                messages.append(msg_data)

                if media_result["has_media"]:
                    media_messages.append(msg_data)

            except Exception:
                continue

        # Try to get profile URL of the other person
        profile_link = await self.page.query_selector(
            ".msg-thread__link-to-profile, .msg-entity-lockup__entity-link"
        )
        other_profile_url = ""
        if profile_link:
            other_profile_url = await profile_link.get_attribute("href") or ""

        return {
            "messages": messages,
            "media_messages": media_messages,
            "has_any_media": len(media_messages) > 0,
            "other_profile_url": other_profile_url,
        }


# ─────────────────────────────────────────────
# MESSAGE HUMANIZER
# ─────────────────────────────────────────────
class Humanizer:
    @staticmethod
    def process(msg: str) -> str:
        if random.random() < 0.25 and len(msg) > 0:
            msg = msg[0].lower() + msg[1:]
        if random.random() < 0.35 and msg.endswith("."):
            msg = msg[:-1]
        replacements = [
            ("I am ", "I'm "), ("do not ", "don't "),
            ("it is ", "it's "), ("that is ", "that's "),
            ("I would ", "I'd "), ("I will ", "I'll "),
            ("cannot ", "can't "),
        ]
        for old, new in replacements:
            if random.random() < 0.6:
                msg = msg.replace(old, new)
        return msg


# ─────────────────────────────────────────────
# PIPELINE ENGINE
# ─────────────────────────────────────────────
class PipelineEngine:
    def __init__(self):
        self.linkedin: Optional[LinkedInBrowser] = None
        self.ai_engine: Optional[AIEngine] = None
        self.multi_ai: Optional[MultiModelAI] = None
        self.running = False
        self.paused = False
        self.loop = None
        self._thread = None
        self.status = "stopped"
        self.last_error = ""
        self.cycle_count = 0

    def get_config(self) -> dict:
        return {
            "linkedin_email": DB.get_config("linkedin_email", os.getenv("LINKEDIN_EMAIL", "")),
            "linkedin_password": DB.get_config("linkedin_password", os.getenv("LINKEDIN_PASSWORD", "")),
            "search_queries": json.loads(
                DB.get_config("search_queries", '["startup founder CEO","CTO co-founder startup","technical founder building"]')
            ),
            "max_connections_per_day": int(DB.get_config("max_connections_per_day", "20")),
            "max_messages_per_day": int(DB.get_config("max_messages_per_day", "40")),
            "max_scrapes_per_day": int(DB.get_config("max_scrapes_per_day", "50")),
            "prospects_per_search": int(DB.get_config("prospects_per_search", "10")),
            "active_start_hour": int(DB.get_config("active_start_hour", "9")),
            "active_end_hour": int(DB.get_config("active_end_hour", "19")),
            "follow_up_after_days": int(DB.get_config("follow_up_after_days", "3")),
            "max_follow_ups": int(DB.get_config("max_follow_ups", "2")),
            "cycle_delay_min": int(DB.get_config("cycle_delay_min", "120")),
            "cycle_delay_max": int(DB.get_config("cycle_delay_max", "300")),
            "dm_check_interval": int(DB.get_config("dm_check_interval", "60")),
            "auto_reply_new_dms": DB.get_config("auto_reply_new_dms", "true") == "true",
            "notify_on_media": DB.get_config("notify_on_media", "true") == "true",
            "your_name": DB.get_config("your_name", ""),
            "your_role": DB.get_config("your_role", ""),
            "your_what_you_do": DB.get_config("your_what_you_do", ""),
            "your_background": DB.get_config("your_background", ""),
            "your_expertise": DB.get_config("your_expertise", ""),
            "your_value_offer": DB.get_config("your_value_offer", ""),
            "your_curiosity": DB.get_config("your_curiosity", ""),
            "target_keywords": json.loads(
                DB.get_config("target_keywords",
                              '["founder","co-founder","ceo","cto","building","startup","builder","maker"]')
            ),
        }

    def get_your_context(self, config: dict) -> dict:
        return {
            "name": config["your_name"],
            "role": config["your_role"],
            "what_you_do": config["your_what_you_do"],
            "background": config["your_background"],
            "expertise": config["your_expertise"],
            "value_offer": config["your_value_offer"],
            "curiosity": config["your_curiosity"],
        }

    def start(self):
        if self.running:
            return
        self.running = True
        self.status = "starting"
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self.running = False
        self.status = "stopping"

    def pause(self):
        self.paused = True
        self.status = "paused"

    def resume(self):
        self.paused = False
        self.status = "running"

    def _run_loop(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)
        try:
            self.loop.run_until_complete(self._main_pipeline())
        except Exception as e:
            logger.error(f"Pipeline crashed: {e}")
            self.last_error = str(e)
            self.status = "error"
        finally:
            self.running = False
            self.status = "stopped"

    async def _main_pipeline(self):
        config = self.get_config()

        # Init AI
        active_provider = DB.get_active_ai_provider()
        if not active_provider:
            self.status = "error"
            self.last_error = "No active AI provider configured. Add one in AI Models tab."
            return

        self.multi_ai = MultiModelAI()
        self.ai_engine = AIEngine(self.multi_ai)

        # Test AI connection
        try:
            ok, msg = self.multi_ai.test_connection(active_provider)
            if not ok:
                self.status = "error"
                self.last_error = f"AI connection failed: {msg}"
                return
        except Exception as e:
            self.status = "error"
            self.last_error = f"AI test failed: {e}"
            return

        # Init browser
        self.linkedin = LinkedInBrowser()
        try:
            await self.linkedin.launch()
            self.status = "logging_in"
            success = await self.linkedin.login(config["linkedin_email"], config["linkedin_password"])
            if not success:
                self.status = "error"
                self.last_error = "LinkedIn login failed"
                return
        except Exception as e:
            self.status = "error"
            self.last_error = f"Browser launch failed: {e}"
            return

        self.status = "running"
        DB.log_activity("pipeline_started",
                        details=f"Provider: {active_provider['provider_name']}/{active_provider['model_name']}")

        last_dm_check = datetime.min

        try:
            while self.running:
                if self.paused:
                    await asyncio.sleep(10)
                    continue

                config = self.get_config()

                if not HumanBehavior.is_active_hours(config):
                    self.status = "sleeping (outside active hours)"
                    await asyncio.sleep(300)
                    continue

                self.status = "running"
                self.cycle_count += 1

                try:
                    today_stats = DB.get_today_stats()

                    # ── 0. DM CHECK (high frequency) ──
                    dm_interval = config.get("dm_check_interval", 60)
                    if (datetime.now() - last_dm_check).total_seconds() >= dm_interval:
                        await self._full_dm_check(config, today_stats)
                        last_dm_check = datetime.now()

                    # ── 1. Handle known incoming messages ──
                    await self._handle_incoming_messages(config, today_stats)

                    # ── 2. Check accepted connections ──
                    await self._handle_accepted_connections(config, today_stats)

                    # ── 3. Follow-ups ──
                    await self._handle_follow_ups(config, today_stats)

                    # ── 4. Send connection requests ──
                    await self._send_connections(config, today_stats)

                    # ── 5. Discover new prospects ──
                    await self._discover_prospects(config, today_stats)

                except Exception as e:
                    logger.error(f"Cycle error: {e}")
                    DB.log_activity("cycle_error", details=str(e), status="error")
                    self.last_error = str(e)

                delay = random.uniform(config["cycle_delay_min"], config["cycle_delay_max"])
                self.status = f"waiting ({int(delay)}s until next cycle)"
                await asyncio.sleep(delay)

        finally:
            await self.linkedin.close()
            DB.log_activity("pipeline_stopped")

    # ────────────────────────────────────────
    # FULL DM CHECK — Checks ALL unread DMs
    # ────────────────────────────────────────
    async def _full_dm_check(self, config: dict, stats: dict):
        """Scan entire inbox for unread messages, including brand new DMs."""
        self.status = "checking all DMs"
        your_context = self.get_your_context(config)

        try:
            unread_threads = await self.linkedin.get_all_unread_threads()
        except Exception as e:
            logger.warning(f"DM inbox check failed: {e}")
            return

        if not unread_threads:
            return

        DB.increment_stat("new_dm_detected", len(unread_threads))

        for thread_info in unread_threads:
            sender_name = thread_info["name"]
            preview = thread_info["preview"]
            thread_url = thread_info.get("thread_url", "")

            # Quick media check on preview
            if thread_info.get("media_detected_preview", {}).get("has_media"):
                media_types = thread_info["media_detected_preview"]["media_types"]
                DB.add_notification(
                    ntype="media_alert",
                    prospect_name=sender_name,
                    message=f"Media in preview: {preview[:100]}",
                    media_type=", ".join(media_types),
                    details=json.dumps(thread_info["media_detected_preview"]),
                )
                DB.increment_stat("media_alerts")
                logger.info(f"📸 Media detected in preview from {sender_name}")

            # Check if this person is a known prospect
            prospect = DB.get_prospect_by_name(sender_name)

            if prospect:
                # Known prospect → handled by _handle_incoming_messages
                logger.info(f"Unread from known prospect: {sender_name}")
                continue

            # ── NEW / UNKNOWN DM ──
            logger.info(f"📩 New DM from unknown sender: {sender_name}")

            # Read full conversation to analyze
            self.status = f"reading new DM: {sender_name}"
            try:
                convo_data = await self.linkedin.read_conversation_with_media_check(thread_url)
            except Exception as e:
                logger.warning(f"Failed to read thread from {sender_name}: {e}")
                continue

            messages = convo_data.get("messages", [])
            media_messages = convo_data.get("media_messages", [])
            other_profile_url = convo_data.get("other_profile_url", "")

            if not messages:
                continue

            latest_msg = messages[-1]

            # ── MEDIA DETECTION ──
            if media_messages or latest_msg.get("has_media"):
                all_media_types = set()
                all_media_details = []
                for mm in media_messages:
                    all_media_types.update(mm.get("media_types", []))
                    all_media_details.extend(mm.get("media_details", []))

                media_type_str = ", ".join(all_media_types) if all_media_types else "unknown"

                DB.add_notification(
                    ntype="media_alert",
                    prospect_name=sender_name,
                    message=(
                        f"{sender_name} sent {media_type_str}. "
                        f"Last text: '{latest_msg.get('text', '')[:200]}'"
                    ),
                    media_type=media_type_str,
                    details=json.dumps({
                        "media_details": all_media_details,
                        "thread_url": thread_url,
                        "profile_url": other_profile_url,
                    }, default=str),
                )
                DB.increment_stat("media_alerts")
                DB.log_activity(
                    "media_received",
                    prospect_name=sender_name,
                    details=f"Media type: {media_type_str}",
                    status="alert",
                )
                logger.info(f"🚨 MEDIA ALERT: {sender_name} sent {media_type_str}")

                # Create prospect entry with NEEDS_HUMAN state
                if other_profile_url:
                    pid = hashlib.md5(other_profile_url.encode()).hexdigest()[:16]
                    DB.upsert_prospect({
                        "id": pid,
                        "name": sender_name,
                        "profile_url": other_profile_url,
                        "state": ProspectState.NEEDS_HUMAN.value,
                        "has_media_pending": True,
                        "conversation_history": json.dumps(
                            [{"sender": "them", "text": m.get("text", ""),
                              "timestamp": m.get("timestamp", ""),
                              "has_media": m.get("has_media", False),
                              "media_types": m.get("media_types", [])}
                             for m in messages[-5:]],
                            default=str
                        ),
                        "notes": f"Media received: {media_type_str}. Needs human review.",
                    })
                continue  # Don't auto-reply to media messages

            # ── AUTO-REPLY TO NEW TEXT DMs ──
            if config.get("auto_reply_new_dms", True):
                their_text = latest_msg.get("text", "")
                if not their_text.strip():
                    continue

                # Try to scrape their profile for context
                sender_profile = {}
                if other_profile_url:
                    try:
                        self.status = f"scraping new contact: {sender_name}"
                        sender_profile = await self.linkedin.scrape_profile(other_profile_url)
                    except Exception:
                        sender_profile = {"name": sender_name}

                # Create prospect in DB
                pid = ""
                if other_profile_url:
                    pid = hashlib.md5(other_profile_url.encode()).hexdigest()[:16]
                    # Try to analyze
                    insight = {}
                    try:
                        insight = self.ai_engine.analyze_profile(sender_profile)
                    except Exception:
                        pass

                    DB.upsert_prospect({
                        "id": pid,
                        "name": sender_name,
                        "profile_url": other_profile_url,
                        "headline": sender_profile.get("headline", ""),
                        "location": sender_profile.get("location", ""),
                        "state": ProspectState.IN_CONVERSATION.value,
                        "profile_data": json.dumps(sender_profile, default=str),
                        "profile_insight": json.dumps(insight, default=str),
                        "source_query": "inbound_dm",
                        "score": insight.get("relevance_score", 50) if insight else 50,
                    })

                # Generate reply
                self.status = f"replying to new DM: {sender_name}"
                try:
                    reply = self.ai_engine.generate_new_dm_reply(
                        sender_name, their_text, sender_profile, your_context
                    )
                    reply = Humanizer.process(reply)
                except Exception as e:
                    logger.warning(f"Reply gen failed for new DM: {e}")
                    DB.add_notification(
                        ntype="human_needed",
                        prospect_name=sender_name,
                        message=f"Couldn't auto-reply. Their msg: {their_text[:200]}",
                    )
                    continue

                # Human-like delay
                await asyncio.sleep(random.uniform(60, 300))

                # Send
                try:
                    if other_profile_url:
                        sent = await self.linkedin.send_message(other_profile_url, reply)
                    else:
                        sent = False
                except Exception:
                    sent = False

                if sent:
                    history = [
                        {"sender": "them", "text": their_text,
                         "timestamp": datetime.utcnow().isoformat()},
                        {"sender": "me", "text": reply,
                         "timestamp": datetime.utcnow().isoformat()},
                    ]
                    if pid:
                        DB.update_prospect(pid, {
                            "conversation_history": json.dumps(history, default=str),
                            "last_message_at": datetime.utcnow(),
                            "last_message_from": "me",
                        })
                    DB.increment_stat("messages_sent")
                    DB.increment_stat("conversations_started")
                    DB.log_activity(
                        "new_dm_replied",
                        prospect_id=pid,
                        prospect_name=sender_name,
                        details=f"Their msg: {their_text[:80]} | Our reply: {reply[:80]}",
                    )
                    logger.info(f"Auto-replied to new DM from {sender_name}")
                else:
                    DB.add_notification(
                        ntype="human_needed",
                        prospect_name=sender_name,
                        message=f"Failed to send auto-reply. Their msg: {their_text[:200]}",
                    )
            else:
                # Auto-reply disabled — just notify
                DB.add_notification(
                    ntype="new_unknown_dm",
                    prospect_name=sender_name,
                    message=f"New DM: {preview[:200]}",
                    details=json.dumps({"thread_url": thread_url}, default=str),
                )

            await HumanBehavior.random_delay(5, 15)

    # ────────────────────────────────────────
    # HANDLE KNOWN PROSPECT MESSAGES
    # ────────────────────────────────────────
    async def _handle_incoming_messages(self, config, stats):
        if stats.get("messages_sent", 0) >= config["max_messages_per_day"]:
            return

        self.status = "checking known prospect messages"
        your_context = self.get_your_context(config)

        active_states = [ProspectState.OPENER_SENT.value, ProspectState.IN_CONVERSATION.value]

        try:
            unread = await self.linkedin.get_all_unread_threads()
        except Exception:
            return

        if not unread:
            return

        for msg_info in unread:
            if stats.get("messages_sent", 0) >= config["max_messages_per_day"]:
                break

            prospect = None
            for state in active_states:
                for p in DB.get_prospects_by_state(state):
                    if (msg_info["name"].lower() in p.get("name", "").lower() or
                            p.get("name", "").lower() in msg_info["name"].lower()):
                        prospect = p
                        break
                if prospect:
                    break

            if not prospect:
                continue  # Handled by _full_dm_check

            # Read full conversation with media check
            self.status = f"reading: {prospect['name']}"
            try:
                convo_data = await self.linkedin.read_conversation_with_media_check(
                    msg_info.get("thread_url", "")
                )
            except Exception:
                convo_data = {"messages": [{"sender": "them", "text": msg_info["preview"]}],
                              "media_messages": [], "has_any_media": False}

            messages = convo_data.get("messages", [])
            media_messages = convo_data.get("media_messages", [])

            if not messages:
                continue

            latest_msg = messages[-1]
            their_message = latest_msg.get("text", "")

            # ── MEDIA CHECK ──
            if latest_msg.get("has_media") or media_messages:
                new_media = [m for m in media_messages if m.get("has_media")]
                if new_media:
                    media_types = set()
                    for mm in new_media:
                        media_types.update(mm.get("media_types", []))

                    media_type_str = ", ".join(media_types)

                    DB.add_notification(
                        ntype="media_alert",
                        prospect_id=prospect["id"],
                        prospect_name=prospect["name"],
                        message=(
                            f"{prospect['name']} sent {media_type_str}. "
                            f"Text: '{their_message[:150]}'"
                        ),
                        media_type=media_type_str,
                        details=json.dumps({
                            "media_details": [d for mm in new_media for d in mm.get("media_details", [])],
                        }),
                    )
                    DB.increment_stat("media_alerts")
                    DB.update_prospect(prospect["id"], {
                        "has_media_pending": True,
                        "state": ProspectState.NEEDS_HUMAN.value,
                        "notes": f"Media received: {media_type_str}. Needs human review.",
                    })
                    DB.log_activity(
                        "media_received",
                        prospect_id=prospect["id"],
                        prospect_name=prospect["name"],
                        details=f"Type: {media_type_str}",
                        status="alert",
                    )
                    logger.info(f"🚨 MEDIA from {prospect['name']}: {media_type_str}")
                    continue  # Don't auto-reply — needs human

            # Text-only media check fallback
            text_media = MediaDetector.detect_media_from_text(their_message)
            if text_media["has_media"]:
                DB.add_notification(
                    ntype="media_alert",
                    prospect_id=prospect["id"],
                    prospect_name=prospect["name"],
                    message=f"Possible media reference: '{their_message[:200]}'",
                    media_type=", ".join(text_media["media_types"]),
                )
                DB.update_prospect(prospect["id"], {"has_media_pending": True})

                # Still try to reply to the text part, but flag it
                if not their_message.strip():
                    continue

            # ── CLASSIFY & REPLY ──
            history = json.loads(prospect.get("conversation_history", "[]"))

            try:
                intent = self.ai_engine.classify_intent(their_message, history)
            except Exception:
                intent = {"sentiment": "neutral", "suggested_action": "reply",
                          "urgency": "normal", "asks_question": False}

            DB.log_activity(
                "message_received",
                prospect_id=prospect["id"],
                prospect_name=prospect["name"],
                details=f"Intent: {intent.get('sentiment')} | {their_message[:100]}",
            )
            DB.increment_stat("replies_received")

            if intent.get("sentiment") == "not_interested":
                DB.update_prospect(prospect["id"], {"state": ProspectState.NOT_INTERESTED.value})
                continue

            if intent.get("suggested_action") == "back_off":
                continue

            delay = min(HumanBehavior.get_reply_delay(intent), 300)
            self.status = f"thinking ({int(delay)}s): {prospect['name']}"
            await asyncio.sleep(delay)

            insight = json.loads(prospect.get("profile_insight", "{}"))
            try:
                reply = self.ai_engine.generate_reply(insight, your_context, history, their_message)
                reply = Humanizer.process(reply)
            except Exception as e:
                logger.warning(f"Reply gen failed: {e}")
                continue

            self.status = f"replying: {prospect['name']}"
            try:
                sent = await self.linkedin.send_message(prospect["profile_url"], reply)
            except Exception:
                sent = False

            if sent:
                history.append({"sender": "them", "text": their_message,
                                "timestamp": datetime.utcnow().isoformat(), "intent": intent})
                history.append({"sender": "me", "text": reply,
                                "timestamp": datetime.utcnow().isoformat()})
                DB.update_prospect(prospect["id"], {
                    "state": ProspectState.IN_CONVERSATION.value,
                    "conversation_history": json.dumps(history, default=str),
                    "last_message_at": datetime.utcnow(),
                    "last_message_from": "me",
                    "follow_up_count": 0,
                })
                DB.increment_stat("messages_sent")
                DB.log_activity("reply_sent", prospect_id=prospect["id"],
                                prospect_name=prospect["name"], details=reply[:200])

            await HumanBehavior.random_delay(30, 90)

    async def _handle_accepted_connections(self, config, stats):
        if stats.get("messages_sent", 0) >= config["max_messages_per_day"]:
            return

        prospects = DB.get_prospects_by_state(ProspectState.CONNECTION_SENT.value)
        your_context = self.get_your_context(config)

        for p in prospects:
            if stats.get("messages_sent", 0) >= config["max_messages_per_day"]:
                break

            self.status = f"checking: {p['name']}"
            try:
                accepted = await self.linkedin.check_connection_accepted(p["profile_url"])
            except Exception:
                continue

            if not accepted:
                sent_at = p.get("connection_sent_at")
                if sent_at:
                    if isinstance(sent_at, str):
                        sent_at = datetime.fromisoformat(sent_at)
                    if datetime.utcnow() - sent_at > timedelta(days=7):
                        DB.update_prospect(p["id"], {"state": ProspectState.NO_RESPONSE.value})
                continue

            DB.update_prospect(p["id"], {
                "state": ProspectState.CONNECTED.value,
                "connection_accepted_at": datetime.utcnow(),
            })
            DB.increment_stat("connections_accepted")

            wait = min(random.uniform(300, 3600), 600)
            self.status = f"waiting {int(wait/60)}m before opener: {p['name']}"
            await asyncio.sleep(wait)

            insight = json.loads(p.get("profile_insight", "{}"))
            try:
                opener = self.ai_engine.generate_opener(insight, your_context)
                opener = Humanizer.process(opener)
            except Exception:
                continue

            self.status = f"opener: {p['name']}"
            try:
                sent = await self.linkedin.send_message(p["profile_url"], opener)
            except Exception:
                continue

            if sent:
                history = [{"sender": "me", "text": opener,
                            "timestamp": datetime.utcnow().isoformat(), "type": "opener"}]
                DB.update_prospect(p["id"], {
                    "state": ProspectState.OPENER_SENT.value,
                    "conversation_history": json.dumps(history),
                    "last_message_at": datetime.utcnow(),
                    "last_message_from": "me",
                })
                DB.increment_stat("messages_sent")
                DB.increment_stat("conversations_started")
                DB.log_activity("opener_sent", prospect_id=p["id"],
                                prospect_name=p["name"], details=opener[:200])

            await HumanBehavior.random_delay(30, 90)

    async def _handle_follow_ups(self, config, stats):
        if stats.get("messages_sent", 0) >= config["max_messages_per_day"]:
            return

        your_context = self.get_your_context(config)
        fu_days = config["follow_up_after_days"]
        max_fu = config["max_follow_ups"]

        for state in [ProspectState.OPENER_SENT.value, ProspectState.IN_CONVERSATION.value]:
            for p in DB.get_prospects_by_state(state):
                if stats.get("messages_sent", 0) >= config["max_messages_per_day"]:
                    return
                if p.get("last_message_from") != "me":
                    continue
                if p.get("follow_up_count", 0) >= max_fu:
                    DB.update_prospect(p["id"], {"state": ProspectState.NO_RESPONSE.value})
                    continue

                last_msg = p.get("last_message_at")
                if not last_msg:
                    continue
                if isinstance(last_msg, str):
                    last_msg = datetime.fromisoformat(last_msg)
                if (datetime.utcnow() - last_msg).days < fu_days:
                    continue

                insight = json.loads(p.get("profile_insight", "{}"))
                history = json.loads(p.get("conversation_history", "[]"))
                attempt = p.get("follow_up_count", 0) + 1

                self.status = f"follow-up #{attempt}: {p['name']}"
                try:
                    msg = self.ai_engine.generate_follow_up(insight, your_context, history, attempt)
                    msg = Humanizer.process(msg)
                except Exception:
                    continue

                try:
                    sent = await self.linkedin.send_message(p["profile_url"], msg)
                except Exception:
                    sent = False

                if sent:
                    history.append({"sender": "me", "text": msg,
                                    "timestamp": datetime.utcnow().isoformat(),
                                    "type": f"follow_up_{attempt}"})
                    DB.update_prospect(p["id"], {
                        "conversation_history": json.dumps(history, default=str),
                        "last_message_at": datetime.utcnow(),
                        "last_message_from": "me",
                        "follow_up_count": attempt,
                    })
                    DB.increment_stat("messages_sent")
                    DB.log_activity(f"follow_up_{attempt}", prospect_id=p["id"],
                                    prospect_name=p["name"], details=msg[:200])

                await HumanBehavior.random_delay(30, 90)

    async def _send_connections(self, config, stats):
        if stats.get("connections_sent", 0) >= config["max_connections_per_day"]:
            return

        prospects = DB.get_prospects_by_state(ProspectState.ANALYZED.value)
        prospects.sort(key=lambda x: float(x.get("score", 0)), reverse=True)
        your_context = self.get_your_context(config)

        for p in prospects[:5]:
            if stats.get("connections_sent", 0) >= config["max_connections_per_day"]:
                break

            insight = json.loads(p.get("profile_insight", "{}"))
            self.status = f"connecting: {p['name']}"
            try:
                note = self.ai_engine.generate_connection_note(insight, your_context)
                note = Humanizer.process(note)
            except Exception:
                note = ""

            try:
                success = await self.linkedin.send_connection_request(p["profile_url"], note)
            except Exception:
                DB.update_prospect(p["id"], {"state": ProspectState.ERROR.value})
                continue

            if success:
                DB.update_prospect(p["id"], {
                    "state": ProspectState.CONNECTION_SENT.value,
                    "connection_note": note,
                    "connection_sent_at": datetime.utcnow(),
                })
                DB.increment_stat("connections_sent")
                DB.log_activity("connection_sent", prospect_id=p["id"],
                                prospect_name=p["name"], details=f"Note: {note[:100]}")
            else:
                DB.update_prospect(p["id"], {"state": ProspectState.ERROR.value,
                                             "error_message": "Connect button not found"})

            await HumanBehavior.random_delay(30, 90)

    async def _discover_prospects(self, config, stats):
        if stats.get("prospects_discovered", 0) >= config["max_scrapes_per_day"]:
            return

        queries = config["search_queries"]
        if not queries:
            return

        query = random.choice(queries)
        self.status = f"searching: {query}"

        try:
            results = await self.linkedin.search_people(query, config["prospects_per_search"])
        except Exception:
            return

        for r in results:
            headline_lower = r.get("headline", "").lower()
            if not any(kw in headline_lower for kw in config.get("target_keywords", [])):
                continue

            pid = hashlib.md5(r["profile_url"].encode()).hexdigest()[:16]
            if DB.get_prospect(pid):
                continue

            self.status = f"scraping: {r['name']}"
            await HumanBehavior.random_delay(2, 5)
            try:
                profile = await self.linkedin.scrape_profile(r["profile_url"])
            except Exception:
                continue

            self.status = f"analyzing: {r['name']}"
            try:
                insight = self.ai_engine.analyze_profile(profile)
            except Exception:
                insight = {}

            DB.upsert_prospect({
                "id": pid, "name": r["name"], "profile_url": r["profile_url"],
                "headline": r.get("headline", ""), "location": r.get("location", ""),
                "state": ProspectState.ANALYZED.value,
                "profile_data": json.dumps(profile, default=str),
                "profile_insight": json.dumps(insight, default=str),
                "source_query": query,
                "score": insight.get("relevance_score", 50),
            })
            DB.increment_stat("prospects_discovered")
            DB.log_activity("prospect_discovered", prospect_id=pid,
                            prospect_name=r["name"],
                            details=f"Score: {insight.get('relevance_score', '?')}")
            await HumanBehavior.random_delay(3, 8)


# ─────────────────────────────────────────────
# STREAMLIT UI
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="LinkedIn Outreach Automation v2",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .stMetric { background: #1a1a2e; padding: 15px; border-radius: 10px; border: 1px solid #16213e; }
    .status-running { color: #00ff88; font-weight: bold; }
    .status-stopped { color: #ff4444; font-weight: bold; }
    .status-paused { color: #ffaa00; font-weight: bold; }
    .conversation-me { background: #1a3a5c; padding: 8px 12px; border-radius: 12px 12px 4px 12px; margin: 4px 0; margin-left: 40px; }
    .conversation-them { background: #2a2a3e; padding: 8px 12px; border-radius: 12px 12px 12px 4px; margin: 4px 0; margin-right: 40px; }
    .notification-badge { background: #ff4444; color: white; border-radius: 50%; padding: 2px 8px; font-size: 0.8em; font-weight: bold; }
    .media-alert { background: #3d1f00; border: 1px solid #ff8c00; padding: 10px; border-radius: 8px; margin: 5px 0; }
    .new-dm-alert { background: #1a3300; border: 1px solid #66ff00; padding: 10px; border-radius: 8px; margin: 5px 0; }
</style>
""", unsafe_allow_html=True)

if "pipeline" not in st.session_state:
    st.session_state.pipeline = PipelineEngine()


def get_pipeline() -> PipelineEngine:
    return st.session_state.pipeline


# ────────────────────────────────────────
# SIDEBAR
# ────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Configuration")

    # Notification count in sidebar
    unread_notifs = DB.get_unread_notifications()
    if unread_notifs:
        st.markdown(
            f'🔔 <span class="notification-badge">{len(unread_notifs)}</span> '
            f"unread notifications",
            unsafe_allow_html=True,
        )

    tab_cred, tab_ai, tab_persona, tab_search, tab_dm = st.tabs(
        ["🔑 Login", "🤖 AI Models", "👤 Persona", "🔍 Search", "💬 DM Settings"]
    )

    with tab_cred:
        st.subheader("LinkedIn Login")
        li_email = st.text_input(
            "Email", value=DB.get_config("linkedin_email", os.getenv("LINKEDIN_EMAIL", "")),
            key="cfg_email"
        )
        li_pass = st.text_input(
            "Password", value=DB.get_config("linkedin_password", os.getenv("LINKEDIN_PASSWORD", "")),
            type="password", key="cfg_pass"
        )
        st.divider()
        st.subheader("Safety Limits")
        max_conn = st.slider("Max connections/day", 5, 40,
                             int(DB.get_config("max_connections_per_day", "20")), key="cfg_mc")
        max_msg = st.slider("Max messages/day", 10, 80,
                            int(DB.get_config("max_messages_per_day", "40")), key="cfg_mm")
        max_scrape = st.slider("Max scrapes/day", 10, 100,
                               int(DB.get_config("max_scrapes_per_day", "50")), key="cfg_ms")
        st.divider()
        st.subheader("Timing")
        c1, c2 = st.columns(2)
        start_h = c1.number_input("Start hour", 0, 23,
                                  int(DB.get_config("active_start_hour", "9")), key="cfg_sh")
        end_h = c2.number_input("End hour", 0, 23,
                                int(DB.get_config("active_end_hour", "19")), key="cfg_eh")
        c3, c4 = st.columns(2)
        cycle_min = c3.number_input("Cycle min(s)", 30, 600,
                                    int(DB.get_config("cycle_delay_min", "120")), key="cfg_cmin")
        cycle_max = c4.number_input("Cycle max(s)", 60, 900,
                                    int(DB.get_config("cycle_delay_max", "300")), key="cfg_cmax")
        fu_days = st.slider("Follow-up after (days)", 1, 10,
                            int(DB.get_config("follow_up_after_days", "3")), key="cfg_fud")
        max_fu = st.slider("Max follow-ups", 0, 5,
                           int(DB.get_config("max_follow_ups", "2")), key="cfg_mfu")

    # ── AI MODELS TAB ──
    with tab_ai:
        st.subheader("🤖 AI Model Providers")
        st.caption("Add multiple providers. Set one as Active and optionally one as Fallback.")

        providers = DB.get_ai_providers()

        # Show existing providers
        for prov in providers:
            emoji = "🟢" if prov.get("is_active") else ("🟡" if prov.get("is_fallback") else "⚪")
            label = prov.get("label") or f"{prov['provider_name']}/{prov['model_name']}"
            with st.expander(f"{emoji} {label}"):
                st.write(f"**Provider:** {prov['provider_name']}")
                st.write(f"**Model:** {prov['model_name']}")
                st.write(f"**Active:** {'✅' if prov.get('is_active') else '❌'}")
                st.write(f"**Fallback:** {'✅' if prov.get('is_fallback') else '❌'}")
                st.write(f"**Temperature:** {prov.get('temperature', 0.85)}")
                st.write(f"**Max Tokens:** {prov.get('max_tokens', 200)}")

                btn_cols = st.columns(4)
                if btn_cols[0].button("Set Active", key=f"act_{prov['id']}"):
                    DB.set_active_provider(prov["id"])
                    st.rerun()
                if btn_cols[1].button("Set Fallback", key=f"fb_{prov['id']}"):
                    with get_db() as db:
                        db.query(AIProviderConfig).update({"is_fallback": False})
                        p = db.query(AIProviderConfig).filter_by(id=prov["id"]).first()
                        if p:
                            p.is_fallback = True
                    st.rerun()
                if btn_cols[2].button("🧪 Test", key=f"test_{prov['id']}"):
                    mai = MultiModelAI()
                    ok, msg = mai.test_connection(prov)
                    if ok:
                        st.success(msg)
                    else:
                        st.error(msg)
                if btn_cols[3].button("🗑️ Delete", key=f"del_prov_{prov['id']}"):
                    DB.delete_ai_provider(prov["id"])
                    st.rerun()

        st.divider()
        st.subheader("➕ Add New Provider")
        with st.form("add_provider"):
            new_provider = st.selectbox(
                "Provider",
                ["openai", "anthropic", "google", "groq"],
                key="new_prov_type",
            )

            available = AVAILABLE_MODELS.get(new_provider, [])
            new_model = st.selectbox("Model", available, key="new_prov_model")
            new_model_custom = st.text_input(
                "Or enter custom model name (overrides above)",
                key="new_prov_custom_model"
            )
            new_key = st.text_input("API Key", type="password", key="new_prov_key")
            new_label = st.text_input(
                "Label (optional)",
                placeholder="e.g. GPT-4o Main",
                key="new_prov_label"
            )
            nc1, nc2 = st.columns(2)
            new_temp = nc1.slider("Temperature", 0.0, 2.0, 0.85, 0.05, key="new_prov_temp")
            new_max_tok = nc2.number_input("Max Tokens", 50, 4096, 200, key="new_prov_mt")
            nc3, nc4 = st.columns(2)
            new_active = nc3.checkbox("Set as Active", key="new_prov_active")
            new_fallback = nc4.checkbox("Set as Fallback", key="new_prov_fb")

            submitted = st.form_submit_button("Add Provider")
            if submitted and new_key:
                final_model = new_model_custom.strip() if new_model_custom.strip() else new_model
                if new_active:
                    with get_db() as db:
                        db.query(AIProviderConfig).update({"is_active": False})
                if new_fallback:
                    with get_db() as db:
                        db.query(AIProviderConfig).update({"is_fallback": False})

                DB.save_ai_provider({
                    "provider_name": new_provider,
                    "api_key": new_key,
                    "model_name": final_model,
                    "is_active": new_active,
                    "is_fallback": new_fallback,
                    "priority": len(providers),
                    "max_tokens": new_max_tok,
                    "temperature": new_temp,
                    "label": new_label or f"{new_provider}/{final_model}",
                })
                st.success(f"Added {new_provider}/{final_model}")
                st.rerun()

        # Supported libraries status
        st.divider()
        st.subheader("Library Status")
        st.write(f"OpenAI: ✅ installed")
        st.write(f"Anthropic: {'✅' if HAS_ANTHROPIC else '❌ pip install anthropic'}")
        st.write(f"Google AI: {'✅' if HAS_GOOGLE else '❌ pip install google-generativeai'}")
        st.write(f"Groq: {'✅' if HAS_GROQ else '❌ pip install groq'}")

    with tab_persona:
        st.subheader("Your Identity")
        your_name = st.text_input("Name", value=DB.get_config("your_name", ""), key="cfg_yn")
        your_role = st.text_input("Role", value=DB.get_config("your_role", ""),
                                  placeholder="e.g. Founder at MyStartup", key="cfg_yr")
        your_do = st.text_area("What You Do", value=DB.get_config("your_what_you_do", ""),
                               height=80, key="cfg_yd")
        your_bg = st.text_area("Background", value=DB.get_config("your_background", ""),
                               height=80, key="cfg_yb")
        your_exp = st.text_area("Expertise", value=DB.get_config("your_expertise", ""),
                                height=60, key="cfg_ye")
        your_val = st.text_area("Value Offer", value=DB.get_config("your_value_offer", ""),
                                height=60, key="cfg_yv")
        your_cur = st.text_area("Curious About", value=DB.get_config("your_curiosity", ""),
                                height=60, key="cfg_yc")

    with tab_search:
        st.subheader("Search Queries")
        queries_raw = st.text_area(
            "One per line",
            value="\n".join(json.loads(
                DB.get_config("search_queries",
                              '["startup founder CEO","CTO co-founder startup","technical founder building"]')
            )),
            height=150, key="cfg_q"
        )
        pps = st.slider("Prospects per search", 3, 25,
                         int(DB.get_config("prospects_per_search", "10")), key="cfg_pps")
        st.divider()
        st.subheader("Target Keywords")
        kw_raw = st.text_area(
            "Filter (one per line)",
            value="\n".join(json.loads(
                DB.get_config("target_keywords",
                              '["founder","co-founder","ceo","cto","building","startup","builder","maker"]')
            )),
            height=120, key="cfg_kw"
        )

    with tab_dm:
        st.subheader("💬 DM Settings")
        dm_interval = st.slider(
            "DM check interval (seconds)", 30, 600,
            int(DB.get_config("dm_check_interval", "60")),
            help="How often to check inbox for new messages",
            key="cfg_dmi"
        )
        auto_reply = st.checkbox(
            "Auto-reply to new/unknown DMs",
            value=DB.get_config("auto_reply_new_dms", "true") == "true",
            help="If someone new DMs you, AI will reply automatically",
            key="cfg_ar"
        )
        notify_media = st.checkbox(
            "Alert on media messages (images/videos/files)",
            value=DB.get_config("notify_on_media", "true") == "true",
            help="Get notified when someone sends non-text content",
            key="cfg_nm"
        )
        st.info(
            "🖼️ When someone sends an image, video, or file, the bot will:\n"
            "1. Detect the media type\n"
            "2. **Pause auto-reply** for that conversation\n"
            "3. **Create a notification** for you to handle manually\n"
            "4. Mark the prospect as 'Needs Human'"
        )

    # SAVE ALL
    if st.button("💾 Save All Settings", use_container_width=True):
        DB.save_config("linkedin_email", li_email)
        DB.save_config("linkedin_password", li_pass)
        DB.save_config("max_connections_per_day", str(max_conn))
        DB.save_config("max_messages_per_day", str(max_msg))
        DB.save_config("max_scrapes_per_day", str(max_scrape))
        DB.save_config("active_start_hour", str(start_h))
        DB.save_config("active_end_hour", str(end_h))
        DB.save_config("cycle_delay_min", str(cycle_min))
        DB.save_config("cycle_delay_max", str(cycle_max))
        DB.save_config("follow_up_after_days", str(fu_days))
        DB.save_config("max_follow_ups", str(max_fu))
        DB.save_config("dm_check_interval", str(dm_interval))
        DB.save_config("auto_reply_new_dms", "true" if auto_reply else "false")
        DB.save_config("notify_on_media", "true" if notify_media else "false")
        DB.save_config("your_name", your_name)
        DB.save_config("your_role", your_role)
        DB.save_config("your_what_you_do", your_do)
        DB.save_config("your_background", your_bg)
        DB.save_config("your_expertise", your_exp)
        DB.save_config("your_value_offer", your_val)
        DB.save_config("your_curiosity", your_cur)
        DB.save_config("prospects_per_search", str(pps))
        queries = [q.strip() for q in queries_raw.strip().split("\n") if q.strip()]
        DB.save_config("search_queries", json.dumps(queries))
        kws = [k.strip().lower() for k in kw_raw.strip().split("\n") if k.strip()]
        DB.save_config("target_keywords", json.dumps(kws))
        st.success("✅ Settings saved!")

# ────────────────────────────────────────
# MAIN AREA
# ────────────────────────────────────────
st.title("🚀 LinkedIn Outreach Automation v2")

pipeline = get_pipeline()

# ── Active AI Provider display ──
active_prov = DB.get_active_ai_provider()
fallback_prov = DB.get_fallback_ai_provider()
ai_status = ""
if active_prov:
    ai_status = f"🤖 **Active:** {active_prov.get('label', active_prov['provider_name']+'/'+active_prov['model_name'])}"
    if fallback_prov:
        ai_status += f" | **Fallback:** {fallback_prov.get('label', fallback_prov['provider_name']+'/'+fallback_prov['model_name'])}"
else:
    ai_status = "⚠️ **No active AI provider configured!** Go to sidebar → AI Models"
st.markdown(ai_status)

# ── Notification Banner ──
unread = DB.get_unread_notifications()
if unread:
    media_alerts = [n for n in unread if n["notification_type"] == "media_alert"]
    dm_alerts = [n for n in unread if n["notification_type"] in ("new_unknown_dm", "human_needed")]

    if media_alerts:
        st.warning(
            f"🖼️ **{len(media_alerts)} media alert(s)** — "
            f"Someone sent images/videos/files. Check the Notifications tab!"
        )
    if dm_alerts:
        st.info(
            f"📩 **{len(dm_alerts)} new DM alert(s)** — "
            f"New messages need your attention. Check Notifications tab!"
        )

# ── CONTROL BAR ──
ctrl_cols = st.columns([1, 1, 1, 1, 3])
with ctrl_cols[0]:
    if st.button("▶️ Start", use_container_width=True, disabled=pipeline.running):
        if not active_prov:
            st.error("Add an AI provider first!")
        else:
            pipeline.start()
            time.sleep(1)
            st.rerun()
with ctrl_cols[1]:
    if pipeline.running and not pipeline.paused:
        if st.button("⏸️ Pause", use_container_width=True):
            pipeline.pause()
            st.rerun()
    elif pipeline.paused:
        if st.button("▶️ Resume", use_container_width=True):
            pipeline.resume()
            st.rerun()
    else:
        st.button("⏸️ Pause", use_container_width=True, disabled=True)
with ctrl_cols[2]:
    if st.button("⏹️ Stop", use_container_width=True, disabled=not pipeline.running):
        pipeline.stop()
        time.sleep(2)
        st.rerun()
with ctrl_cols[3]:
    if st.button("🔄 Refresh", use_container_width=True):
        st.rerun()
with ctrl_cols[4]:
    status = pipeline.status
    if pipeline.running and not pipeline.paused:
        st.markdown(f'**Status:** <span class="status-running">🟢 {status}</span>', unsafe_allow_html=True)
    elif pipeline.paused:
        st.markdown(f'**Status:** <span class="status-paused">🟡 PAUSED</span>', unsafe_allow_html=True)
    else:
        st.markdown(f'**Status:** <span class="status-stopped">🔴 {status}</span>', unsafe_allow_html=True)

if pipeline.last_error:
    st.error(f"Last error: {pipeline.last_error}")

st.divider()

# ── TABS ──
tab_dash, tab_notif, tab_prospects, tab_convos, tab_activity, tab_manual = st.tabs([
    "📊 Dashboard",
    f"🔔 Notifications ({len(unread)})" if unread else "🔔 Notifications",
    "👥 Prospects",
    "💬 Conversations",
    "📋 Activity Log",
    "🛠️ Manual Actions",
])

# ── DASHBOARD ──
with tab_dash:
    stats = DB.get_today_stats()
    all_prospects = DB.get_all_prospects()

    st.subheader("📈 Today's Stats")
    m_cols = st.columns(8)
    m_cols[0].metric("Discovered", stats.get("prospects_discovered", 0))
    m_cols[1].metric("Conn Sent", stats.get("connections_sent", 0))
    m_cols[2].metric("Accepted", stats.get("connections_accepted", 0))
    m_cols[3].metric("Msgs Sent", stats.get("messages_sent", 0))
    m_cols[4].metric("Replies", stats.get("replies_received", 0))
    m_cols[5].metric("Convos", stats.get("conversations_started", 0))
    m_cols[6].metric("📸 Media Alerts", stats.get("media_alerts", 0))
    m_cols[7].metric("📩 New DMs", stats.get("new_dm_detected", 0))

    st.divider()
    st.subheader("🔄 Pipeline")
    state_counts = {}
    for p in all_prospects:
        s = p.get("state", "unknown")
        state_counts[s] = state_counts.get(s, 0) + 1

    funnel = [
        ("discovered", "🔍"), ("analyzed", "🧠"), ("connection_sent", "📤"),
        ("connected", "🤝"), ("opener_sent", "💬"), ("in_conversation", "🗣️"),
        ("qualified", "⭐"), ("needs_human", "👤"), ("not_interested", "❌"),
        ("no_response", "😶"), ("error", "⚠️"),
    ]
    f_cols = st.columns(len(funnel))
    for i, (sk, em) in enumerate(funnel):
        f_cols[i].metric(f"{em}", state_counts.get(sk, 0))

    # Chart
    history = DB.get_stats_history(30)
    if history:
        import pandas as pd
        df = pd.DataFrame(history)
        if not df.empty:
            df["date"] = pd.to_datetime(df["date"])
            chart_cols = ["connections_sent", "messages_sent", "replies_received", "media_alerts"]
            existing = [c for c in chart_cols if c in df.columns]
            if existing:
                st.line_chart(df.set_index("date")[existing])

# ── NOTIFICATIONS ──
with tab_notif:
    st.subheader("🔔 Notifications")

    if unread:
        if st.button("✅ Mark All as Read"):
            DB.mark_all_notifications_read()
            st.rerun()

    all_notifs = DB.get_all_notifications(200)
    if not all_notifs:
        st.info("No notifications yet.")
    else:
        for n in all_notifs:
            ts = n.get("timestamp", "")
            if isinstance(ts, datetime):
                ts = ts.strftime("%m/%d %H:%M")
            is_unread = not n.get("is_read", False)
            ntype = n.get("notification_type", "")

            if ntype == "media_alert":
                icon = "🖼️"
                css_class = "media-alert"
            elif ntype == "new_unknown_dm":
                icon = "📩"
                css_class = "new-dm-alert"
            elif ntype == "human_needed":
                icon = "👤"
                css_class = "media-alert"
            else:
                icon = "🔔"
                css_class = ""

            badge = " **[NEW]**" if is_unread else ""

            with st.expander(
                f"{icon} {n.get('prospect_name','')} — "
                f"{n.get('notification_type','')} — {ts}{badge}"
            ):
                st.markdown(
                    f'<div class="{css_class}">{n.get("message", "")}</div>',
                    unsafe_allow_html=True,
                )

                if n.get("media_type"):
                    st.write(f"**Media Type:** {n['media_type']}")

                details = n.get("details", "{}")
                if details and details != "{}":
                    try:
                        det = json.loads(details) if isinstance(details, str) else details
                        if det.get("thread_url"):
                            st.write(f"**Thread:** {det['thread_url']}")
                        if det.get("media_details"):
                            st.write(f"**Details:** {', '.join(det['media_details'][:5])}")
                        if det.get("profile_url"):
                            st.write(f"**Profile:** [{det['profile_url']}]({det['profile_url']})")
                    except Exception:
                        st.write(details)

                nc1, nc2, nc3 = st.columns(3)
                if nc1.button("✅ Mark Read", key=f"nr_{n['id']}"):
                    DB.mark_notification_read(n["id"])
                    st.rerun()
                if nc2.button("✅ Resolve", key=f"nres_{n['id']}"):
                    DB.mark_notification_resolved(n["id"])
                    st.rerun()
                if n.get("prospect_id"):
                    if nc3.button("👤 View Prospect", key=f"nvp_{n['id']}"):
                        st.session_state["view_prospect"] = n["prospect_id"]

# ── PROSPECTS ──
with tab_prospects:
    st.subheader("👥 All Prospects")
    filter_state = st.selectbox("Filter", ["all"] + [s.value for s in ProspectState], key="fs")

    all_p = DB.get_all_prospects()
    if filter_state != "all":
        all_p = [p for p in all_p if p.get("state") == filter_state]

    st.caption(f"Showing {len(all_p)} prospects")

    for p in all_p[:100]:
        state_emoji = {
            "discovered": "🔍", "analyzed": "🧠", "connection_sent": "📤",
            "connected": "🤝", "opener_sent": "💬", "in_conversation": "🗣️",
            "qualified": "⭐", "needs_human": "👤", "not_interested": "❌",
            "no_response": "😶", "error": "⚠️",
        }
        em = state_emoji.get(p.get("state", ""), "❓")
        media_flag = " 🖼️" if p.get("has_media_pending") else ""

        with st.expander(
            f"{em} **{p.get('name','')}**{media_flag} — "
            f"{p.get('headline','')[:60]} | Score: {p.get('score', 0)}"
        ):
            ca, cb = st.columns(2)
            with ca:
                st.write(f"**State:** {p.get('state','')}")
                st.write(f"**Location:** {p.get('location','')}")
                st.write(f"**Profile:** [{p.get('profile_url','')}]({p.get('profile_url','')})")
                st.write(f"**Source:** {p.get('source_query','')}")
                if p.get("has_media_pending"):
                    st.warning("🖼️ Media message pending — needs human review")
                if p.get("notes"):
                    st.write(f"**Notes:** {p['notes']}")
            with cb:
                insight = json.loads(p.get("profile_insight", "{}"))
                if insight:
                    st.write(f"**Building:** {insight.get('what_they_are_building','?')}")
                    st.write(f"**Stage:** {insight.get('company_stage','?')}")
                    st.write(f"**Pain Points:** {', '.join(insight.get('pain_points', []))}")

            history = json.loads(p.get("conversation_history", "[]"))
            if history:
                st.write("---")
                st.write("**Conversation:**")
                for msg in history:
                    sender = msg.get("sender", "?")
                    text = msg.get("text", "")
                    ts_str = str(msg.get("timestamp", ""))[:16]
                    media_tag = ""
                    if msg.get("has_media"):
                        media_tag = f" 🖼️ [{', '.join(msg.get('media_types', []))}]"
                    if sender == "me":
                        st.markdown(
                            f'<div class="conversation-me"><small>You • {ts_str}</small><br>{text}{media_tag}</div>',
                            unsafe_allow_html=True,
                        )
                    else:
                        st.markdown(
                            f'<div class="conversation-them"><small>{p.get("name","")} • {ts_str}</small><br>{text}{media_tag}</div>',
                            unsafe_allow_html=True,
                        )

            mc1, mc2, mc3, mc4 = st.columns(4)
            if mc1.button("⭐ Qualified", key=f"q_{p['id']}"):
                DB.update_prospect(p["id"], {"state": ProspectState.QUALIFIED.value})
                st.rerun()
            if mc2.button("❌ Not Interested", key=f"ni_{p['id']}"):
                DB.update_prospect(p["id"], {"state": ProspectState.NOT_INTERESTED.value})
                st.rerun()
            if mc3.button("🔄 Reset to Convo", key=f"rc_{p['id']}"):
                DB.update_prospect(p["id"], {
                    "state": ProspectState.IN_CONVERSATION.value,
                    "has_media_pending": False,
                })
                st.rerun()
            if mc4.button("🗑️ Delete", key=f"d_{p['id']}"):
                with get_db() as db:
                    db.query(ProspectModel).filter_by(id=p["id"]).delete()
                st.rerun()

# ── CONVERSATIONS ──
with tab_convos:
    st.subheader("💬 Active Conversations")
    convo_states = [
        ProspectState.OPENER_SENT.value,
        ProspectState.IN_CONVERSATION.value,
        ProspectState.QUALIFIED.value,
        ProspectState.NEEDS_HUMAN.value,
    ]
    convo_prospects = [p for p in DB.get_all_prospects() if p.get("state") in convo_states]

    if not convo_prospects:
        st.info("No active conversations yet.")
    else:
        for p in convo_prospects:
            history = json.loads(p.get("conversation_history", "[]"))
            needs_human = p.get("state") == ProspectState.NEEDS_HUMAN.value
            media_flag = " 🖼️ **MEDIA — NEEDS HUMAN**" if needs_human else ""

            st.write(f"### {p.get('name','')}{media_flag}")
            if needs_human:
                st.warning(
                    "This conversation has media content that needs your review. "
                    "Reply manually and click 'Reset to Convo' when done."
                )

            for msg in history:
                sender = msg.get("sender", "?")
                text = msg.get("text", "")
                ts_str = str(msg.get("timestamp", ""))[:16]
                media_tag = ""
                if msg.get("has_media"):
                    media_tag = f" 🖼️ [{', '.join(msg.get('media_types', []))}]"
                if sender == "me":
                    st.markdown(
                        f'<div class="conversation-me"><small>You • {ts_str}</small><br>{text}{media_tag}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div class="conversation-them"><small>{p.get("name","")} • {ts_str}</small><br>{text}{media_tag}</div>',
                        unsafe_allow_html=True,
                    )

            manual_msg = st.text_input("Send manual message", key=f"mm_{p['id']}")
            if st.button("Send", key=f"sm_{p['id']}"):
                if manual_msg:
                    history.append({
                        "sender": "me", "text": manual_msg,
                        "timestamp": datetime.utcnow().isoformat(), "type": "manual",
                    })
                    DB.update_prospect(p["id"], {
                        "conversation_history": json.dumps(history, default=str),
                        "last_message_at": datetime.utcnow(),
                        "last_message_from": "me",
                    })
                    DB.log_activity("manual_message_queued",
                                    prospect_id=p["id"], prospect_name=p.get("name", ""),
                                    details=manual_msg[:200])
                    st.success("Message saved (will be sent in next cycle)")
                    st.rerun()
            st.divider()

# ── ACTIVITY LOG ──
with tab_activity:
    st.subheader("📋 Recent Activity")
    activities = DB.get_recent_activity(100)
    if not activities:
        st.info("No activity yet.")
    else:
        for act in activities:
            ts = act.get("timestamp", "")
            if isinstance(ts, datetime):
                ts = ts.strftime("%m/%d %H:%M")
            color = "log-success" if act.get("status") == "success" else "log-error"
            icon = "📸" if "media" in act.get("action", "") else "●"
            st.markdown(
                f'`{ts}` {icon} **{act.get("action","")}** — '
                f'{act.get("prospect_name","")} '
                f'<small>{str(act.get("details",""))[:120]}</small>',
                unsafe_allow_html=True,
            )

# ── MANUAL ACTIONS ──
with tab_manual:
    st.subheader("🛠️ Manual Actions")

    st.write("### Add Prospect")
    with st.form("add_p"):
        m_name = st.text_input("Name")
        m_url = st.text_input("Profile URL")
        m_hl = st.text_input("Headline")
        if st.form_submit_button("Add") and m_name and m_url:
            pid = hashlib.md5(m_url.encode()).hexdigest()[:16]
            DB.upsert_prospect({
                "id": pid, "name": m_name, "profile_url": m_url,
                "headline": m_hl, "state": ProspectState.DISCOVERED.value,
            })
            st.success(f"Added: {m_name}")

    st.divider()
    st.write("### Bulk Import CSV")
    uploaded = st.file_uploader("CSV (name, profile_url, headline)", type="csv")
    if uploaded:
        import pandas as pd
        try:
            df = pd.read_csv(uploaded)
            count = 0
            for _, row in df.iterrows():
                name = str(row.get("name", ""))
                url = str(row.get("profile_url", ""))
                if name and url:
                    pid = hashlib.md5(url.encode()).hexdigest()[:16]
                    DB.upsert_prospect({
                        "id": pid, "name": name, "profile_url": url,
                        "headline": str(row.get("headline", "")),
                        "state": ProspectState.DISCOVERED.value,
                    })
                    count += 1
            st.success(f"Imported {count}")
        except Exception as e:
            st.error(str(e))

    st.divider()
    st.write("### AI Quick Test")
    test_prompt = st.text_input("Test prompt", value="Say hello in a casual way")
    if st.button("🧪 Test Active AI"):
        try:
            mai = MultiModelAI()
            result = mai.chat(
                [{"role": "user", "content": test_prompt}],
                max_tokens=100, temperature=0.8
            )
            st.success(f"Response: {result}")
        except Exception as e:
            st.error(f"Error: {e}")

    st.divider()
    st.write("### Database")
    dc1, dc2, dc3 = st.columns(3)
    if dc1.button("🗑️ Clear Prospects"):
        with get_db() as db:
            db.query(ProspectModel).delete()
        st.rerun()
    if dc2.button("🗑️ Clear Logs"):
        with get_db() as db:
            db.query(ActivityLog).delete()
        st.rerun()
    if dc3.button("🗑️ Clear Notifications"):
        with get_db() as db:
            db.query(NotificationModel).delete()
        st.rerun()
    if st.button("📥 Export JSON"):
        st.download_button(
            "Download", json.dumps(DB.get_all_prospects(), indent=2, default=str),
            "prospects.json", "application/json",
        )
    st.divider()
    st.write("### Reset State")
    rc1, rc2 = st.columns(2)
    rf = rc1.selectbox("From", [s.value for s in ProspectState], key="rf")
    rt = rc2.selectbox("To", [s.value for s in ProspectState], key="rt")
    if st.button("Reset"):
        ps = DB.get_prospects_by_state(rf)
        for p in ps:
            DB.update_prospect(p["id"], {"state": rt})
        st.success(f"Reset {len(ps)}")
        st.rerun()

    st.divider()
    if Path(LOG_FILE).exists():
        st.text_area("Log File", Path(LOG_FILE).read_text()[-5000:], height=200, disabled=True)

# ── AUTO-REFRESH ──
if pipeline.running:
    st.markdown("""
        <script>setTimeout(function(){window.location.reload();}, 30000);</script>
    """, unsafe_allow_html=True)
