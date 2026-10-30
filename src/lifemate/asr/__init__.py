"""ASR integration interfaces."""
from .base import ASR
from .phowhisper import ASRError, PhoWhisperASR
__all__ = ['ASR', 'ASRError', 'PhoWhisperASR']
