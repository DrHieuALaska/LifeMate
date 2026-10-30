"""Contract checks without models, GPU, private data or optional dependencies."""
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest
from unittest.mock import Mock, patch

from lifemate.asr import ASRError, PhoWhisperASR


class AdapterTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.audio = Path(self.temp.name) / "synthetic.wav"
        self.audio.write_bytes(b"synthetic fixture, backend is mocked")
        self.asr = PhoWhisperASR.__new__(PhoWhisperASR)
        self.asr._backend = Mock()

    def test_raw_transcript_preserved_and_only_audio_passed(self):
        self.asr._backend.transcribe.return_value = {
            "status": "ok", "text_model": "  Xin CHÀO!  ",
            "text_normalized": "xin chào", "latency_ms": 5,
        }
        self.assertEqual(self.asr.transcribe(self.audio), "  Xin CHÀO!  ")
        self.asr._backend.transcribe.assert_called_once_with(self.audio)

    def test_partial_and_empty_outputs_are_errors(self):
        for result in (
            {"status": "error", "text_model": "private partial transcript"},
            {"status": "ok", "text_model": " "},
            {"status": "ok", "text_model": None},
            {"text_model": "missing status"},
        ):
            with self.subTest(result=result):
                self.asr._backend.transcribe.return_value = result
                with self.assertRaises(ASRError) as caught:
                    self.asr.transcribe(self.audio)
                self.assertNotIn("private", str(caught.exception))

    def test_missing_audio_does_not_invoke_backend(self):
        with self.assertRaises(FileNotFoundError):
            self.asr.transcribe(self.audio.parent / "missing.wav")
        self.asr._backend.transcribe.assert_not_called()

    def test_missing_checkpoint_rejected_before_loading(self):
        with self.assertRaises(FileNotFoundError):
            PhoWhisperASR(self.audio.parent / "missing-model")

    def test_checkpoint_loaded_once_and_cpu_gpu_defaults(self):
        from lifemate.asr.phowhisper import MODEL_ID, MODEL_REVISION
        for device, precision in (("cpu", "float32"), ("cuda:0", "float16")):
            with self.subTest(device=device):
                backend_module = types.ModuleType("lifemate.asr.asr_benchmark")
                backend_module.ASR = Mock()
                backend_module.ASR.return_value.transcribe.return_value = {
                    "status": "ok", "text_model": "synthetic text"
                }
                with patch.dict(sys.modules, {backend_module.__name__: backend_module}):
                    adapter = PhoWhisperASR(self.audio.parent, device=device)
                backend_module.ASR.assert_called_once_with(
                    spec={"model_id": MODEL_ID, "family": "seq2seq"},
                    source=self.audio.parent.resolve(), revision=MODEL_REVISION,
                    device=device, precision=precision, max_new_tokens=400,
                )
                adapter.transcribe(self.audio)
                adapter.transcribe(self.audio)
                self.assertEqual(backend_module.ASR.call_count, 1)

    def test_import_does_not_load_optional_ml_stack(self):
        code = (
            "import sys; from lifemate.asr import ASR, PhoWhisperASR; "
            "assert not {'torch', 'transformers', 'pandas', 'numpy'} & sys.modules.keys()"
        )
        subprocess.run([sys.executable, "-c", code], check=True)


if __name__ == "__main__":
    unittest.main()
