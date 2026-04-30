import os
from dotenv import load_dotenv
load_dotenv() # Load first!

from sqlalchemy import text
from plutchik_erc_dashboard.database import SessionLocal, engine, Base

def test_connection():
    print(f"Testing connection to: {engine.url}")
    try:
        # Create tables
        Base.metadata.create_all(bind=engine)
        print("✓ Tables created successfully.")
        
        # Test query
        with engine.connect() as conn:
            result = conn.execute(text("SELECT 1"))
            print(f"✓ Query result: {result.fetchone()[0]}")
            
        print("🚀 Local Postgres is fully integrated with Plutchik ERC!")
    except Exception as e:
        print(f"❌ Connection failed: {e}")

if __name__ == "__main__":
    test_connection()
