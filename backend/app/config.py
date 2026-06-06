from pydantic_settings import BaseSettings
from typing import List

class Settings(BaseSettings):
    DATABASE_URL:          str   = "postgresql://reli:secret@localhost:5432/reli_db"
    REDIS_URL:             str   = "redis://localhost:6379"
    SECRET_KEY:            str   = "dev-secret-key-replace-in-production"
    MAX_FACES_PER_CAMERA:  int   = 35
    EMOTION_SKIP_FRAMES:   int   = 5
    METRIC_WRITE_INTERVAL: int   = 30
    ALERT_THRESHOLD:       float = 50.0
    ALERT_COOLDOWN:        int   = 120
    CORS_ORIGINS:          List[str] = ["http://localhost:3000"]

    class Config:
        env_file = ".env"

settings = Settings()
