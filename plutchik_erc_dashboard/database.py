import os
from sqlalchemy import create_engine
from sqlalchemy.orm import declarative_base, sessionmaker

# 1. Resolve Data Path (Priority: Environment Variable > HF Persistent Volume > Local)
PERSISTENT_PATH = os.getenv("PERSISTENT_DATA_PATH", ".")
DEFAULT_SQLITE_DB = os.path.join(PERSISTENT_PATH, "plutchik_erc.db")

# 2. Resolve Database URL
DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_SQLITE_DB}")

# 3. Handle Supabase/Heroku dialect fix (postgres -> postgresql)
if DATABASE_URL.startswith("postgres://"):
    DATABASE_URL = DATABASE_URL.replace("postgres://", "postgresql://", 1)

# 3.1 Force SSL for remote Supabase/Postgres connections
if "localhost" not in DATABASE_URL and "127.0.0.1" not in DATABASE_URL and "sqlite" not in DATABASE_URL:
    if "sslmode" not in DATABASE_URL:
        separator = "&" if "?" in DATABASE_URL else "?"
        DATABASE_URL += f"{separator}sslmode=require"

# 4. Engine Configuration (Pooling for production)
engine_args = {
    "pool_pre_ping": True, # Ensure connections are alive
    "pool_size": 5,        # Baseline for HF Space
    "max_overflow": 10     # Burst capacity
}

if "sqlite" in DATABASE_URL:
    engine_args = {"connect_args": {"check_same_thread": False}}

# 5. Create engine with fallback to SQLite if PostgreSQL is unavailable
try:
    engine = create_engine(DATABASE_URL, **engine_args)
    # Quick connectivity test for PostgreSQL
    if "sqlite" not in DATABASE_URL:
        engine.connect().close()
except Exception as e:
    print(f"⚠ PostgreSQL unavailable ({e}). Falling back to SQLite at {DEFAULT_SQLITE_DB}")
    DATABASE_URL = f"sqlite:///{DEFAULT_SQLITE_DB}"
    engine_args = {"connect_args": {"check_same_thread": False}}
    engine = create_engine(DATABASE_URL, **engine_args)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
