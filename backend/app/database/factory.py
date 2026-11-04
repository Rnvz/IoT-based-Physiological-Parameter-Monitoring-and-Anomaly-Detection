from app.config import settings
from app.database.interfaces import BaseRepository
from app.database.csv_repository import CSVRepository
from app.database.supabase_repository import SupabaseRepository

_repo_instance: BaseRepository = None

def get_repository() -> BaseRepository:
    global _repo_instance
    if _repo_instance is None:
        backend_name = (settings.db_backend or "csv").strip().lower()
        if backend_name in ("supabase", "dual"):
            _repo_instance = SupabaseRepository(
                supabase_url=settings.supabase_url,
                supabase_service_key=settings.supabase_service_key,
                csv_dir=settings.csv_dir
            )
        else:
            _repo_instance = CSVRepository(csv_dir=settings.csv_dir)
    return _repo_instance
