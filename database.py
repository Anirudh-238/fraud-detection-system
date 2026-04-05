"""
database.py — SQLAlchemy setup + User table definition.

Creates a local SQLite database file: fraudshield.db
Tables:
    users       — registered accounts
    login_logs  — every login attempt (for velocity tracking)
"""

from sqlalchemy import create_engine, Column, String, Integer, Boolean, Float, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime, timezone

# ─────────────────────────────────────────────
# Database setup
# ─────────────────────────────────────────────

DATABASE_URL = "sqlite:///./fraudshield.db"

engine = create_engine(
    DATABASE_URL,
    connect_args={"check_same_thread": False}  # needed for SQLite + FastAPI
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


# ─────────────────────────────────────────────
# Tables
# ─────────────────────────────────────────────

class User(Base):
    """Registered user account."""
    __tablename__ = "users"

    id                  = Column(Integer, primary_key=True, index=True)
    email               = Column(String, unique=True, index=True, nullable=False)
    full_name           = Column(String, nullable=False)
    hashed_password     = Column(String, nullable=False)

    # Signup metadata
    created_at          = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    signup_ip           = Column(String, nullable=True)
    signup_country      = Column(String, nullable=True)
    device_id           = Column(String, nullable=True)

    # Updated after every successful login
    last_login_at       = Column(DateTime, nullable=True)
    last_known_ip       = Column(String, nullable=True)
    last_known_country  = Column(String, nullable=True)

    # Fraud tracking
    failed_attempts     = Column(Integer, default=0)
    is_flagged          = Column(Boolean, default=False)   # REVIEW flag
    is_blocked          = Column(Boolean, default=False)   # hard BLOCK flag


class LoginLog(Base):
    """Every login attempt — used for velocity and history."""
    __tablename__ = "login_logs"

    id           = Column(Integer, primary_key=True, index=True)
    email        = Column(String, index=True)
    ip_address   = Column(String)
    country      = Column(String, nullable=True)
    user_agent   = Column(String, nullable=True)
    device_id    = Column(String, nullable=True)
    fraud_score  = Column(Float, default=0.0)
    action       = Column(String, default="allow")   # allow / review / block
    timestamp    = Column(DateTime, default=lambda: datetime.now(timezone.utc))


# ─────────────────────────────────────────────
# Init — call this once at startup
# ─────────────────────────────────────────────

def init_db():
    """Create all tables if they don't exist yet."""
    Base.metadata.create_all(bind=engine)
    print("  [DB] Database ready — fraudshield.db")


# ─────────────────────────────────────────────
# Dependency — use in FastAPI route functions
# ─────────────────────────────────────────────

def get_db():
    """Yields a database session, closes it when done."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()