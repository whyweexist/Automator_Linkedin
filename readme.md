# LinkedIn Outreach Automation Dashboard

An AI-powered automation and analytics dashboard for managing LinkedIn prospecting workflows, tracking outreach performance, and generating personalized messages at scale.

Built for fast iteration and real-time insights using Streamlit and SQLAlchemy.

---

## 🚀 Features

### 🤖 AI Message Generation

* Personalized LinkedIn connection and follow-up messages
* Powered by Gemini models via the modern SDK
* Configurable prompt templates

### 📊 Outreach Analytics Dashboard

* Daily prospect discovery tracking
* Connection requests and acceptance metrics
* Reply and conversation monitoring
* Automated daily statistics storage

### 🗄 Database-Driven Tracking

* SQLite persistence
* Daily stats aggregation
* Schema auto-upgrade support
* SQLAlchemy ORM integration

### ⚡ Streamlit Interface

* Real-time metrics display
* Interactive dashboard
* Stateless UI with safe DB session lifecycle

---

## 🧱 Tech Stack

* Python 3.12+
* Streamlit
* SQLAlchemy ORM
* SQLite
* Gemini AI via Google GenAI SDK

---

## 📦 Installation

### 1️⃣ Clone the repository

```bash
git clone [<your-repo-url>](https://github.com/whyweexist/Automator_Linkedin)
cd Automator_Linkedin
```

### 2️⃣ Install dependencies

```bash
pip install -r requirements.txt
```

If installing manually:

```bash
pip install streamlit playwright openai anthropic google-generativeai groq sqlalchemy apscheduler
```

---

## 🔑 Environment Setup

Create a `.env` file:

```
GOOGLE_API_KEY=your_api_key_here
```

---

## ▶️ Running the Application

```bash
streamlit run app.py
```

The dashboard will open in your browser.

---

## 🧠 AI Integration

The app uses the new Gemini SDK.

Example usage:

```python
from google import genai

client = genai.Client(api_key=GOOGLE_API_KEY)

response = client.models.generate_content(
    model="gemini-1.5-flash",
    contents="Write a professional LinkedIn connection request",
)

print(response.text)
```

Recommended model for automation:

* gemini-1.5-flash → fast, cost-efficient
* gemini-1.5-pro → advanced reasoning

---

## 🗄 Database Behavior

The application automatically:

* Creates tables on first run
* Ensures required columns exist
* Tracks daily metrics
* Uses timezone-aware UTC dates

Daily statistics tracked:

* prospects_discovered
* connections_sent
* messages_sent
* replies_received
* conversations_started
* media_alerts
* new_dm_detected

---

## ⚙️ Architecture Overview

```
Streamlit UI
     ↓
Service Layer (automation + AI)
     ↓
Database Layer (SQLAlchemy ORM)
     ↓
SQLite Storage
```

Key design principles:

* Stateless UI rendering
* Safe session-scoped DB access
* Primitive value extraction before UI display
* Schema compatibility guards

---

## 🛠 Common Issues & Fixes

### DetachedInstanceError

Cause: ORM object accessed after session close
Fix: Extract primitive values before rendering UI

### Missing DB Columns

Cause: Model updated after table creation
Fix: App auto-adds missing columns at startup

### datetime.utcnow Deprecation

Fixed by using:

```python
from datetime import datetime, UTC
datetime.now(UTC)
```

---

## 📈 Future Improvements

* Alembic migrations
* Background automation workers
* Multi-account support
* CSV export for analytics
* PostgreSQL production backend
* Outreach campaign management

---

## 🧑‍💻 Author

Built as part of a full-stack automation workflow system combining:

* AI generation
* Analytics dashboarding
* Persistent tracking infrastructure

