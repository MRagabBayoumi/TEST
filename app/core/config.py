import os
from pathlib import Path
from dotenv import load_dotenv

# Base Project Directory
BASE_DIR = Path(__file__).resolve().parent.parent.parent

# Load .env file
load_dotenv(BASE_DIR / ".env")


class Settings:
    PROJECT_NAME: str = os.getenv("PROJECT_NAME", "Tender AI Lab")
    VERSION: str = os.getenv("VERSION", "0.2.0")
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() in ("true", "1", "yes")

    # Storage paths (resolved to absolute paths)
    BASE_DIR: Path = BASE_DIR
    UPLOAD_DIR: Path = BASE_DIR / os.getenv("UPLOAD_DIR", "storage/uploads")
    PROCESSED_DIR: Path = BASE_DIR / os.getenv("PROCESSED_DIR", "storage/processed")
    LOGS_DIR: Path = BASE_DIR / os.getenv("LOGS_DIR", "storage/logs")
    TESSDATA_DIR: Path = BASE_DIR / os.getenv("TESSDATA_DIR", "storage/tessdata")

    # Security limits
    MAX_UPLOAD_SIZE_MB: int = int(os.getenv("MAX_UPLOAD_SIZE_MB", 250))
    ALLOWED_EXTENSIONS: list[str] = ["pdf"]

    # OCR Settings
    TESSERACT_CMD: str = os.getenv(
        "TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.exe"
    )
    OCR_DPI: int = int(os.getenv("OCR_DPI", 200))
    OCR_LANGUAGES: str = os.getenv("OCR_LANGUAGES", "ara+eng")

    def __init__(self):
        # Ensure storage directories exist
        self.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        self.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        self.LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.TESSDATA_DIR.mkdir(parents=True, exist_ok=True)


settings = Settings()
