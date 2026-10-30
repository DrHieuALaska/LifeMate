"""Local PhoWhisper adapter for the team's ASR protocol.

Importing this module does not import the ML stack or download model weights.
"""
from pathlib import Path


MODEL_ID = "vinai/PhoWhisper-medium"
MODEL_REVISION = "55a7e3eb6c906de891f8f06a107754427dd3be79"


class ASRError(RuntimeError):
    """Transcription failed; the caller may ask the user to repeat the audio."""


class PhoWhisperASR:
    """Load an explicitly supplied local checkpoint once, then transcribe audio.

    Provision MODEL_ID at MODEL_REVISION separately. A local path alone cannot
    prove its upstream revision; callers own checkpoint provenance. Audio must
    be mono 16 kHz and at most 30 seconds, as in the reviewed benchmark.
    """

    def __init__(self, checkpoint_path: Path, *, device: str = "cpu",
                 precision: str | None = None):
        checkpoint = Path(checkpoint_path).resolve()
        if not checkpoint.is_dir():
            raise FileNotFoundError(f"Local ASR checkpoint directory missing: {checkpoint}")
        from .asr_benchmark import ASR as BenchmarkASR

        self._backend = BenchmarkASR(
            spec={"model_id": MODEL_ID, "family": "seq2seq"},
            source=checkpoint, revision=MODEL_REVISION, device=device,
            precision=precision or ("float16" if device.startswith("cuda") else "float32"),
            max_new_tokens=400,
        )

    def transcribe(self, audio_path: Path) -> str:
        """Return raw model text, never scoring-normalized text or partial errors."""
        path = Path(audio_path)
        if not path.is_file():
            raise FileNotFoundError(f"Audio file missing: {path}")
        result = self._backend.transcribe(path)
        text = result.get("text_model")
        if result.get("status") != "ok" or not isinstance(text, str) or not text.strip():
            # Do not expose model output, private file paths or transcript in errors.
            raise ASRError("Local ASR did not produce a complete transcript")
        return text
