import os
import json
import random
import threading
from app.config import settings

class SequenceManager:
    """
    Manages a 32-bit integer counter for /cmd messages, persisted in a JSON file.
    Starts at a random integer, increments by 1 per command, never uses millisecond timestamps.
    """

    def __init__(self, filepath: str = settings.cmd_seq_file):
        self.filepath = os.path.abspath(filepath)
        os.makedirs(os.path.dirname(self.filepath), exist_ok=True)
        self._lock = threading.Lock()
        self.current_seq = self._load_or_init()

    def _load_or_init(self) -> int:
        if os.path.exists(self.filepath):
            try:
                with open(self.filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    return int(data.get("seq", random.randint(1000, 50000)))
            except Exception:
                pass
        
        initial_seq = random.randint(1000, 50000)
        self._save(initial_seq)
        return initial_seq

    def _save(self, seq: int):
        try:
            with open(self.filepath, "w", encoding="utf-8") as f:
                json.dump({"seq": seq}, f)
        except Exception as e:
            print(f"[SequenceManager] Error saving sequence counter: {e}")

    def next_seq(self) -> int:
        with self._lock:
            self.current_seq += 1
            if self.current_seq > 2147483647:
                self.current_seq = 1
            self._save(self.current_seq)
            return self.current_seq

_seq_manager_instance = None

def get_seq_manager() -> SequenceManager:
    global _seq_manager_instance
    if _seq_manager_instance is None:
        _seq_manager_instance = SequenceManager()
    return _seq_manager_instance
