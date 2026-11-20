"""
Revision R4 extension to the prototype: compute KronoDroid-compatible *static* features from an APK file, so that
the Android model can be applied to APKs recovered from a device (adb backup with -apk, /data/app/*/base.apk).

Only features with an unambiguous definition are produced (the 'deployable static subset'):
  - one binary flag per Android permission name used by KronoDroid (short names, e.g. READ_SMS)
  - nr_permissions (number of requested permissions), CFileSize (APK bytes), UFileSize (sum of uncompressed entry
    sizes), FilesInsideAPK, Activities, NrServices
Uses androguard (pure Python).
"""
import os, zipfile
from androguard.core.apk import APK

SIMPLE_COUNTS = ["nr_permissions", "CFileSize", "UFileSize", "FilesInsideAPK", "Activities", "NrServices"]


def deployable_columns(all_static_cols):
    perms = [c for c in all_static_cols if c.isupper() and not c.startswith("SYS_")]
    return perms + [c for c in SIMPLE_COUNTS if c in all_static_cols]


def apk_features(path, columns):
    a = APK(path)
    perms = {p.split(".")[-1] for p in a.get_permissions()}
    with zipfile.ZipFile(path) as z:
        infos = z.infolist()
    f = {c: (1 if c in perms else 0) for c in columns if c not in SIMPLE_COUNTS}
    f.update(nr_permissions=len(set(a.get_permissions())), CFileSize=os.path.getsize(path),
             UFileSize=sum(i.file_size for i in infos), FilesInsideAPK=len(infos),
             Activities=len(a.get_activities()), NrServices=len(a.get_services()))
    meta = dict(package=a.get_package(), version=a.get_androidversion_name(), min_sdk=a.get_min_sdk_version(),
                n_permissions_unmapped=len(perms - set(columns)))
    return {c: f.get(c, 0) for c in columns}, meta
