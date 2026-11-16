"""Atomic local artifacts and optional whole-run checkpoint mirrors."""
import json
import os
from pathlib import Path
import shutil
import tempfile


def write_json_exclusive(path, value):
    path=Path(path)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'w',encoding='utf-8') as stream:
            json.dump(value,stream,ensure_ascii=False,indent=2,allow_nan=False)
            stream.write('\n');stream.flush();os.fsync(stream.fileno())
        # Atomic publication without replacing any already published artifact.
        os.link(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


def write_bytes_exclusive(path, data):
    path=Path(path)
    fd,temporary=tempfile.mkstemp(prefix=path.name+'.',suffix='.tmp',dir=path.parent)
    try:
        with os.fdopen(fd,'wb') as stream:
            stream.write(data);stream.flush();os.fsync(stream.fileno())
        os.link(temporary,path)
    finally:
        if os.path.exists(temporary):os.unlink(temporary)


class RunMirror:
    """Copy completed files, keeping transient/partial files out of the backup."""
    def __init__(self, source, target=None):
        self.source=Path(source).resolve();self.target=Path(target).resolve() if target else None
        self.copied={}
        if self.target and (self.target.is_relative_to(self.source) or self.source.is_relative_to(self.target)):
            raise ValueError('backup and output directories must be separate')

    def __call__(self):
        if self.target is None:return
        self.target.mkdir(parents=True,exist_ok=True)
        for source in self.source.rglob('*'):
            if not source.is_file() or source.name.endswith('.tmp'):continue
            relative=source.relative_to(self.source);stat=source.stat()
            signature=(stat.st_mtime_ns,stat.st_size)
            if self.copied.get(relative)==signature:continue
            dest=self.target/relative;dest.parent.mkdir(parents=True,exist_ok=True)
            temporary=dest.with_name(dest.name+'.tmp')
            shutil.copyfile(source,temporary);os.replace(temporary,dest)
            self.copied[relative]=signature
