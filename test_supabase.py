import os
from supabase import create_client
from dotenv import load_dotenv

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")

print(f"Supabase URL: {SUPABASE_URL}")
print(f"Supabase Key: {SUPABASE_KEY}")

try:
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    # Test the connection
    result = supabase.table("emotions").select("id").limit(1).execute()
    print("Connection successful!")
    print(result)
except Exception as e:
    print(f"Connection failed: {e}")