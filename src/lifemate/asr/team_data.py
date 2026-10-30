"""Read ASR inputs using the team's existing development manifests.

No split generation, environment access, public-test reads or tool-label use.
The historical 991-audio cache remains a separate experiment.
"""
from dataclasses import dataclass, field
from pathlib import Path

from lifemate.evaluation.dataset import read_ids
from lifemate.utils.json import loads
from lifemate.utils.safety import safe_path


@dataclass(frozen=True)
class AudioInput:
    sample_id: str
    kind: str
    audio_path: Path
    # Reference is for offline scoring only; never pass it into ASR inference.
    reference: str = field(repr=False)


def load_dev_audio(data_root: Path, manifest: Path, *, expected_count: int = 193) -> list[AudioInput]:
    """Return current/user-history audio in the original dev manifest order.

    Reuse the shared manifest reader and path guard. Missing audio/text raises
    instead of silently shrinking the evaluation denominator. This does not
    load models, resample audio, create splits or return raw dataset records.
    """
    root, manifest = safe_path(data_root), safe_path(manifest)
    if manifest.name != "dev_ids.txt":
        raise ValueError("ASR development requires the team's existing dev_ids.txt")
    dev = read_ids(manifest)
    if len(dev) != expected_count:
        raise ValueError(f"Expected {expected_count} development IDs; found {len(dev)}")
    partitions = {"dev": set(dev)}
    for name in ("train", "holdout"):
        path = manifest.parent / f"{name}_ids.txt"
        if path.is_file():
            partitions[name] = set(read_ids(path))
    names = list(partitions)
    for i, left in enumerate(names):
        for right in names[i + 1:]:
            if partitions[left] & partitions[right]:
                raise ValueError("Train/dev/holdout manifests overlap")

    def audio_input(record, sid, kind):
        audio = record.get("audio")
        text = record.get("audio_text", record.get("text"))
        if not isinstance(audio, str) or not isinstance(text, str) or not text.strip():
            raise ValueError("Selected audio has missing path/reference; repair preprocessing first")
        path = safe_path(root / "train" / audio)
        if path.suffix.lower() != ".wav" or not path.is_relative_to((root / "train/audio").resolve()):
            raise ValueError("Selected audio must stay under train/audio")
        if not path.is_file():
            raise FileNotFoundError("Selected audio file missing; restore the team's dataset")
        return AudioInput(sid, kind, path, text)

    chosen, seen = {}, set()
    source = safe_path(root / "train/train.jsonl")
    with source.open(encoding="utf-8-sig") as stream:
        for line in stream:
            if not line.strip():
                continue
            record = loads(line)
            if not isinstance(record, dict) or not isinstance(record.get("id"), str):
                raise ValueError("Training record requires a string ID")
            sid = record["id"]
            if sid in seen:
                raise ValueError("Duplicate training sample ID")
            seen.add(sid)
            if sid not in partitions["dev"]:
                continue
            rows = [audio_input(record, sid, "final")]
            for turn in record.get("history", []):
                role = turn.get("role")
                if role in ("user", "human") and turn.get("audio"):
                    rows.append(audio_input(turn, sid, "history"))
                elif turn.get("audio") and role is None:
                    raise ValueError("Historical audio requires an explicit role")
            chosen[sid] = rows
    if any(ids - seen for ids in partitions.values()):
        raise ValueError("Manifest contains IDs absent from train/train.jsonl")
    return [audio for sid in dev for audio in chosen[sid]]
