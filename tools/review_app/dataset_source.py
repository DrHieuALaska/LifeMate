"""Read the original package from a ZIP or an already extracted directory."""
from pathlib import Path, PurePosixPath
import zipfile


class DatasetSource:
    def __init__(self, path):
        self.path = Path(path).expanduser().resolve()
        self.directory = self.path.is_dir()
        self._files = {}
        if self.directory:
            for item in self.path.rglob('*'):
                if item.is_file() and item.resolve().is_relative_to(self.path):
                    self._files[item.relative_to(self.path).as_posix()] = item
            self._names = list(self._files)
        elif self.path.is_file() and zipfile.is_zipfile(self.path):
            with zipfile.ZipFile(self.path) as archive:
                self._names = archive.namelist()
            if len(self._names) != len(set(self._names)):
                raise ValueError('Duplicate member paths in dataset ZIP')
        else:
            raise FileNotFoundError('Dataset must be an original ZIP or extracted package directory.')
        self._name_set = set(self._names)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def namelist(self):
        return list(self._names)

    def manifest(self, split):
        filename = split + '.jsonl'
        candidates = [name for name in self._names
                      if '__MACOSX' not in PurePosixPath(name).parts
                      and (name == filename or PurePosixPath(name).parts[-2:] == (split, filename))]
        if len(candidates) != 1:
            raise ValueError(f'Expected one {filename}; select the exact package or {split} directory.')
        return candidates[0]

    def resolve_member(self, member):
        relative = PurePosixPath(member)
        if relative.is_absolute() or '..' in relative.parts or '\\' in member or ':' in member:
            raise ValueError('Unsafe dataset member path')
        if member in self._name_set:
            return member
        # A valid snapshot can retain a ZIP wrapper prefix; resolve the same
        # train/public_test member against the current extracted package layout.
        for split in ('train', 'public_test'):
            if split not in relative.parts:
                continue
            index = relative.parts.index(split)
            suffix = relative.parts[index + 1:]
            try:
                parent = PurePosixPath(self.manifest(split)).parent
            except ValueError:
                continue
            candidate = str(parent.joinpath(*suffix))
            if candidate in self._name_set:
                return candidate
        raise FileNotFoundError('Dataset file not found: ' + member)

    def read(self, member):
        name = self.resolve_member(member)
        if self.directory:
            path = self._files[name].resolve()
            if not path.is_relative_to(self.path):
                raise ValueError('Dataset path escapes selected directory')
            return path.read_bytes()
        with zipfile.ZipFile(self.path) as archive:
            return archive.read(name)
