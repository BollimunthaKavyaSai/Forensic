"""
Pure-Python stand-in for the pytsk3 and pyewf interfaces used by the prototype's extraction modules.

The native pytsk3/pyewf DLLs are blocked by the Application Control policy of the evaluation machine, so this
module provides the small subset of their API that the prototype calls (Img_Info, Volume_Info, FS_Info, fs.open,
fs.open_dir, File.read_random, entry.info.name/meta), implemented with Fox-IT's dissect.evidence (EWF) and
dissect.ntfs. The prototype's artefact parsers themselves are run unchanged.

Usage: import tsk_shim; tsk_shim.install()   # before importing the prototype's extraction package
"""
import sys, types, glob as _glob, os, struct
from datetime import timezone
from dissect.evidence.ewf import EWF
from dissect.ntfs import NTFS


# ------------------------------------------------------------------ pyewf
class _EwfHandle:
    def open(self, files):
        self._fhs = [open(f, "rb") for f in sorted(files)]
        self._s = EWF(self._fhs).open()
        self._s.seek(0, 2); self._size = self._s.tell(); self._s.seek(0)

    def seek(self, off): self._s.seek(off)
    def read(self, n): return self._s.read(n)
    def get_media_size(self): return self._size
    def close(self): [f.close() for f in self._fhs]


def _ewf_glob(path):
    base = path[:-4] if path.lower().endswith(".e01") else path
    return sorted(_glob.glob(_glob.escape(base) + ".E[0-9][0-9]"))


pyewf = types.ModuleType("pyewf"); pyewf.handle = _EwfHandle; pyewf.glob = _ewf_glob


# ------------------------------------------------------------------ pytsk3
class Img_Info:
    def __init__(self, url="", type=None, **kw): pass
    def read(self, offset, size): raise NotImplementedError
    def get_size(self): raise NotImplementedError


class _Part:
    def __init__(self, start, length, desc): self.start, self.len, self.desc, self.addr = start, length, desc, 0


def Volume_Info(img):
    mbr = img.read(0, 512)
    if mbr[510:512] != b"\x55\xaa": raise IOError("no MBR")
    parts = []
    for i in range(4):
        e = mbr[446 + 16 * i: 462 + 16 * i]; ptype = e[4]; start, n = struct.unpack("<II", e[8:16])
        if ptype and n:
            parts.append(_Part(start, n, (f"NTFS / exFAT (0x{ptype:02x})" if ptype == 7 else f"0x{ptype:02x}").encode()))
    return parts


class _OffsetFH:
    def __init__(self, img, base): self.img, self.base, self.pos = img, base, 0
    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.img.get_size() - self.base + off
        return self.pos
    def tell(self): return self.pos
    def read(self, n=-1):
        if n is None or n < 0: n = self.img.get_size() - self.base - self.pos
        b = self.img.read(self.base + self.pos, n); self.pos += len(b); return b


def _ts(dt):
    try: return int(dt.replace(tzinfo=timezone.utc).timestamp()) if dt.tzinfo is None else int(dt.timestamp())
    except Exception: return 0


class _Meta:
    def __init__(self, rec):
        self.size = rec.size() if rec.is_file() else 0
        self.type = 2 if rec.is_dir() else 1  # TSK_FS_META_TYPE_REG = 1, DIR = 2
        si = None
        try: si = rec.attributes[0x10][0]
        except Exception: pass
        self.crtime = _ts(si.creation_time) if si else 0
        self.mtime = _ts(si.last_modification_time) if si else 0
        self.atime = _ts(si.last_access_time) if si else 0
        self.ctime = _ts(si.last_change_time) if si else 0


class _Name:
    def __init__(self, name): self.name = name.encode("utf-8", "surrogatepass")


class _Info:
    def __init__(self, name, rec): self.name = _Name(name); self.meta = _Meta(rec)


class File:
    def __init__(self, rec, name):
        self._rec = rec; self.info = _Info(name, rec)
    def read_random(self, offset, size, *a):
        s = self._rec.open(); s.seek(offset); return s.read(size)
    def as_directory(self): return _Dir(self._rec)


class _Dir:
    def __init__(self, rec): self._rec = rec
    def __iter__(self):
        for name, entry in self._rec.listdir().items():
            try: yield File(entry.dereference(), name)
            except Exception: continue


class FS_Info:
    def __init__(self, img, offset=0, **kw):
        self._ntfs = NTFS(_OffsetFH(img, offset))
    def _get(self, path):
        p = path.replace("\\", "/").strip("/")
        return self._ntfs.mft.get(p) if p else self._ntfs.mft.root
    def open(self, path):
        rec = self._get(path); return File(rec, os.path.basename(path.rstrip("/")) or "/")
    def open_dir(self, path="/", **kw):
        rec = self._get(path)
        if not rec.is_dir(): raise IOError(f"not a directory: {path}")
        return _Dir(rec)


pytsk3 = types.ModuleType("pytsk3")
for k, v in dict(Img_Info=Img_Info, Volume_Info=Volume_Info, FS_Info=FS_Info, TSK_IMG_TYPE_EXTERNAL=0,
                 TSK_FS_META_TYPE_REG=1, TSK_FS_META_TYPE_DIR=2, TSK_FS_NAME_TYPE_DIR=3, TSK_FS_NAME_TYPE_REG=5).items():
    setattr(pytsk3, k, v)


def install():
    sys.modules["pytsk3"] = pytsk3; sys.modules["pyewf"] = pyewf
