from app.config import settings
from app.database.interfaces import BaseRepository
from app.database.csv_repository import CSVRepository

_repo_instance: BaseRepository = None

def get_repository() -> BaseRepository:
    global _repo_instance
    if _repo_instance is None:
        if settings.db_backend.lower() == "csv":
            _repo_instance = CSVRepository(csv_dir=settings.csv_dir)
        else:
            # Fallback to CSV if other backend not yet configured
            _repo_instance = CSVRepository(csv_dir=settings.csv_dir)
    return _repo_instance
