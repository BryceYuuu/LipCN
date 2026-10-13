"""Portable-archive regressions with a tiny fixture; no native builds or imports."""
from pathlib import Path
import shutil
import stat
import os
import zipfile

import pytest

from scripts import build_lipcn


def test_archive_keeps_bundle_tree_and_executable_modes(tmp_path):
    app = tmp_path/'LipCN.app'
    executable = app/'Contents/MacOS/LipCN'
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b'fixture executable')
    executable.chmod(0o755)
    (app/'Contents/Resources/empty').mkdir(parents=True)
    destination = tmp_path/'release.zip'
    build_lipcn.archive_bundle(app, destination)
    with zipfile.ZipFile(destination) as archive:
        assert archive.read('LipCN.app/Contents/MacOS/LipCN') == b'fixture executable'
        mode = archive.getinfo('LipCN.app/Contents/MacOS/LipCN').external_attr >> 16
        # Windows chmod does not implement POSIX execute bits. The ZIP must
        # preserve the actual source mode; macOS/Linux must also retain +x.
        assert stat.S_IMODE(mode) == stat.S_IMODE(executable.stat().st_mode)
        if os.name != 'nt':
            assert stat.S_IMODE(mode) == 0o755
        assert archive.getinfo('LipCN.app/Contents/Resources/empty/').is_dir()
        assert archive.testzip() is None
    assert not destination.with_name('release.zip.part').exists()


def test_low_disk_aborts_only_new_partial_preserving_app_and_previous_release(tmp_path, monkeypatch):
    app = tmp_path/'LipCN.app'
    app.mkdir()
    (app/'fixture').write_bytes(b'payload')
    destination = tmp_path/'release.zip'
    destination.write_bytes(b'previous release')
    usage = shutil.disk_usage(tmp_path)
    monkeypatch.setattr(build_lipcn.shutil, 'disk_usage', lambda _: type(usage)(usage.total, usage.used, 512*1024*1024))
    with pytest.raises(OSError, match='free space'):
        build_lipcn.archive_bundle(app, destination)
    assert destination.read_bytes() == b'previous release'
    assert (app/'fixture').read_bytes() == b'payload'
    assert not destination.with_name('release.zip.part').exists()


def test_copy_filter_removes_editable_private_paths_but_keeps_licenses():
    names = ['direct_url.json', '_editable_impl_lipcn.pth', '_virtualenv.py', '__pycache__',
             'LICENSE.txt', 'licenses', 'METADATA', 'model.bin']
    assert set(build_lipcn.ignored('.', names)) == set(names[:4])


def test_collection_reference_files_are_part_of_packaged_research_tree():
    root = Path(__file__).resolve().parents[1]
    assert (root/'research/SILENT_EVALUATION.md').is_file()
    assert len(list((root/'research/examples').glob('silent-prompts-*.json'))) == 3
