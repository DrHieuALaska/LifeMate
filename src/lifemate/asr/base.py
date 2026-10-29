from pathlib import Path
from typing import Protocol


class ASR(Protocol):
    def transcribe(self, audio_path: Path) -> str:
        """Transcribe authorized current or historical audio locally."""
        ...
