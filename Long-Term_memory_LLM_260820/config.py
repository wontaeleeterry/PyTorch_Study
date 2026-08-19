from pathlib import Path


# ============================================================
# Project paths
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
DATA_DIR.mkdir(exist_ok=True)


# ============================================================
# LM Studio
# ============================================================

LM_STUDIO_BASE_URL = "http://localhost:1234/v1"

LM_STUDIO_API_KEY = "lm-studio"

# LM Studio에서 실제 로드한 모델명으로 변경
MODEL_NAME = "google/gemma-4-26b-a4b-qat"


# ============================================================
# Embedding
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"


# ============================================================
# Conversation
# ============================================================

# 최근 대화 몇 개를 LLM context에 직접 전달할 것인가
RECENT_TURNS = 8


# ============================================================
# Retrieval
# ============================================================

# FAISS에서 검색할 장기기억 수
TOP_K = 5

# similarity threshold
# 너무 낮은 score의 memory는 사용하지 않음
MEMORY_SCORE_THRESHOLD = 0.25


# ============================================================
# Summary
# ============================================================

# 몇 turn마다 memory consolidation 수행
SUMMARY_INTERVAL = 5

# summary 생성 시 사용할 최근 conversation
SUMMARY_SOURCE_TURNS = 20


# ============================================================
# Storage
# ============================================================

SQLITE_DB = DATA_DIR / "conversation.db"

FAISS_INDEX = DATA_DIR / "faiss.index"

MEMORY_METADATA = DATA_DIR / "memories.pkl"