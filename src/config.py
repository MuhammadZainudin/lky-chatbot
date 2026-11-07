import os
from dotenv import load_dotenv

load_dotenv()

EMBED_MODEL = "all-MiniLM-L6-v2"
DB_PATH = "db"
COLLECTION = "lky"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 120
TOP_K = 5
LLM_MODEL = os.getenv("LKY_MODEL", "claude-sonnet-4-6")
JUDGE_MODEL = os.getenv("LKY_JUDGE_MODEL", "claude-sonnet-4-6")
