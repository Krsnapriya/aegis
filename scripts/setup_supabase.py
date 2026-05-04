import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from sqlalchemy import create_engine
from plutchik_erc_dashboard.database import Base
from plutchik_erc_dashboard.models.db_models import DB_Prediction, DB_Correction, DB_DialogueTurn

def setup_db(db_url):
    print(f"Connecting to: {db_url.split('@')[-1]} (password masked)")
    
    # Supabase/Heroku fix
    if db_url.startswith("postgres://"):
        db_url = db_url.replace("postgres://", "postgresql://", 1)
        
    engine = create_engine(db_url)
    
    print("Creating tables if they do not exist...")
    Base.metadata.create_all(bind=engine)
    print("✅ Tables created successfully!")

if __name__ == "__main__":
    # Priority: Argument > Environment Variable
    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = os.getenv("DATABASE_URL")
        
    if not url:
        print("Error: No DATABASE_URL provided.")
        print("Usage: python scripts/setup_supabase.py 'postgresql://postgres:password@db.xyz.supabase.co:5432/postgres'")
        sys.exit(1)
        
    setup_db(url)
