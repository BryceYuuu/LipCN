"""Offline preparation contracts: no backend, device or public model is exercised."""
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import types

import pytest

from lipcn import assets, cli, setup


def test_manifest_pins_every_public_asset_and_never_publishes_local_adapter(tmp_path):
    items = assets.plan('all', environment={'HOME': str(tmp_path)}, wording=True)
    assert len(items) >= 20
    assert len({item.path for item in items}) == len(items)
    for item in items:
        assert item.path.is_relative_to(tmp_path)
        if item.url:
            assert item.url.startswith('https://')
            assert len(item.sha256) == 64 and item.size > 0 and item.revision
            assert '/main/' not in item.url
        else:
            assert item.key == 'chinese:local-adapter' and item.research_license
    assert not list(tmp_path.iterdir())


def test_asset_modes_respect_user_overrides_and_include_desktop_preview_asset(tmp_path):
    items = assets.plan('speech', environment={'HOME': str(tmp_path), 'LIPCN_ASR_MODEL': str(tmp_path/'asr')})
    assert {item.group for item in items} == {'asr', 'face'}
    assert not any(item.research_license for item in items)
    assert all(item.path.parent == tmp_path/'asr' for item in items if item.group == 'asr')
    assert {item.group for item in assets.plan('english')} == {'english', 'face'}


@pytest.mark.parametrize('mode', assets.MODES)
def test_wording_model_is_opt_in_for_every_mode(tmp_path, mode):
    env = {'HOME': str(tmp_path)}
    assert 'wording' not in {item.group for item in assets.plan(mode, environment=env)}
    with_wording = assets.plan(mode, environment=env, wording=True)
    assert 'wording' in {item.group for item in with_wording}
    assert len(with_wording) == len(assets.plan(mode, environment=env)) + sum(
        item.group == 'wording' for item in with_wording)


def test_english_ready_without_optional_wording_model(tmp_path, monkeypatch):
    env = {'HOME': str(tmp_path)}
    for item in assets.plan('english', environment=env):
        item.path.parent.mkdir(parents=True, exist_ok=True)
        item.path.write_bytes(b'x' * (item.size or 1))
    monkeypatch.setattr(setup.importlib.metadata, 'version', lambda name: '1')
    monkeypatch.setattr(setup.platform, 'system', lambda: 'Darwin')
    monkeypatch.setattr(setup.platform, 'machine', lambda: 'arm64')
    monkeypatch.setattr(setup.platform, 'python_version_tuple', lambda: ('3', '12', '0'))
    monkeypatch.setattr(setup.platform, 'mac_ver', lambda: ('14.0', ('', '', ''), ''))
    report = setup.check('english', environment=env)
    assert report['ready'] and report['wording'] is False
    assert not setup.check('english', environment=env, wording=True)['ready']


@pytest.mark.parametrize('argv,env_mode,expected', [
    (['doctor'], None, False),
    (['doctor', '--with-wording'], None, True),
    (['doctor'], 'model', True),
    (['setup', '--mode', 'speech'], None, False),
    (['setup', '--mode', 'speech', '--with-wording'], None, True),
    (['setup', '--mode', 'speech'], 'model', True),
])
def test_cli_wording_flag_and_model_mode_include_wording(monkeypatch, argv, env_mode, expected):
    seen = []
    report = {'ready': True, 'assets': [], 'dependencies': []}
    monkeypatch.setattr(setup, 'check', lambda *a, **k: seen.append(k['wording']) or report)
    monkeypatch.setattr(setup, 'prepare', lambda *a, **k: seen.append(k['wording']) or report)
    if env_mode is None:
        monkeypatch.delenv('LIPCN_WORDING_MODE', raising=False)
    else:
        monkeypatch.setenv('LIPCN_WORDING_MODE', env_mode)
    assert cli.main(argv + ['--json']) == 0
    assert seen == [expected]


def test_doctor_is_stdlib_only_and_does_not_create_profile_or_cache(tmp_path):
    root = Path(__file__).resolve().parents[1]
    program = """
import json, sys, urllib.request
urllib.request.urlopen = lambda *a, **k: (_ for _ in ()).throw(AssertionError('network'))
from lipcn.setup import check
report = check('speech', environment={'HOME': sys.argv[1]})
assert not any(name in sys.modules for name in ('torch','cv2','mlx.core','sounddevice','AppKit','mediapipe'))
assert not report['ready']
print(json.dumps(report))
"""
    result = subprocess.run([sys.executable, '-B', '-S', '-c', program, str(tmp_path)], cwd=root,
                            text=True, capture_output=True, check=True)
    assert json.loads(result.stdout)['schema_version'] == 1
    assert not list(tmp_path.iterdir())


class Response(io.BytesIO):
    def __init__(self, body, status=200, headers=None):
        super().__init__(body)
        self.status, self.headers = status, headers or {}


def asset(tmp_path, body=b'0123456789'):
    return assets.Asset('test', 'asr', tmp_path/'model.bin', 'https://example.invalid/pinned/file',
                        hashlib.sha256(body).hexdigest(), len(body))


def test_resumes_partial_with_range_and_atomically_publishes_verified_bytes(tmp_path):
    item = asset(tmp_path)
    partial = item.path.with_name('model.bin.lipcn-part')
    partial.write_bytes(b'0123')
    def opener(request, **kwargs):
        assert request.headers['Range'] == 'bytes=4-'
        return Response(b'456789', 206, {'Content-Range': 'bytes 4-9/10'})
    setup.download(item, opener=opener)
    assert item.path.read_bytes() == b'0123456789' and not partial.exists()


def test_server_ignoring_range_restarts_instead_of_appending(tmp_path):
    item = asset(tmp_path)
    item.path.with_name('model.bin.lipcn-part').write_bytes(b'bad partial')
    setup.download(item, opener=lambda *a, **k: Response(b'0123456789'))
    assert item.path.read_bytes() == b'0123456789'


def test_hash_failure_preserves_previous_model_and_clears_corrupt_partial(tmp_path):
    item = asset(tmp_path)
    item.path.write_bytes(b'previous valid user file')
    with pytest.raises(ValueError, match='SHA256'):
        setup.download(item, opener=lambda *a, **k: Response(b'XXXXXXXXXX'))
    assert item.path.read_bytes() == b'previous valid user file'
    assert not item.path.with_name('model.bin.lipcn-part').exists()


def test_interrupted_download_leaves_resumable_partial(tmp_path):
    item = asset(tmp_path)
    with pytest.raises(ValueError, match='Incomplete'):
        setup.download(item, opener=lambda *a, **k: Response(b'0123'))
    assert not item.path.exists()
    assert item.path.with_name('model.bin.lipcn-part').read_bytes() == b'0123'


@pytest.mark.parametrize('mode', ['mandarin', 'english', 'all'])
def test_research_download_requires_acceptance_before_any_network(monkeypatch, mode):
    calls = []
    monkeypatch.setattr(setup, 'download', lambda *a, **k: calls.append(a))
    with pytest.raises(ValueError, match='explicit acceptance'):
        setup.prepare(mode, download=True)
    assert not calls


def test_default_setup_only_checks_and_never_downloads(monkeypatch):
    monkeypatch.setattr(setup, 'download', lambda *a, **k: pytest.fail('unexpected download'))
    monkeypatch.setattr(setup, 'check', lambda *a, **k: {'ready': False})
    assert setup.prepare('english') == {'ready': False}


def test_verified_existing_file_never_opens_network(tmp_path):
    item = asset(tmp_path)
    item.path.write_bytes(b'0123456789')
    setup.download(item, opener=lambda *a, **k: pytest.fail('unexpected network'))


def test_cli_defaults_to_new_desktop_and_not_legacy(monkeypatch):
    calls = []
    monkeypatch.setattr(cli, 'launch', lambda: calls.append('current-desktop') or 0)
    assert cli.main([]) == 0 and cli.main(['run']) == 0
    assert calls == ['current-desktop', 'current-desktop']


def test_desktop_entry_is_included_in_package_layout():
    assert cli.desktop_path().is_file()
    assert (cli.desktop_path().parent/'config.py').is_file()


def test_doctor_cli_has_no_dependency_on_legacy_native_entry(monkeypatch, capsys):
    report = {'ready': False, 'assets': [], 'dependencies': []}
    monkeypatch.setattr(setup, 'check', lambda *a, **k: report)
    assert cli.main(['doctor', '--mode', 'english', '--json']) == 1
    assert json.loads(capsys.readouterr().out) == report


def test_collect_forwards_arguments_only_on_explicit_command(monkeypatch, tmp_path):
    calls = []
    module = types.ModuleType('research.scripts.collect_silent_chinese')
    module.main = lambda args: calls.append(args) or 3
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('LIPCN_PROFILE_DIR', str(tmp_path/'profile'))
    monkeypatch.setenv('LIPFLOW_HOME', str(tmp_path/'legacy'))
    assert cli.main(['collect', '--help']) == 3
    assert calls == [['--help']]
    assert cli.os.environ['LIPFLOW_HOME'] == str(tmp_path/'profile')


@pytest.mark.parametrize('command,module_name', [('audit','audit_silent_data'),('report','report_silent_errors')])
def test_collection_analysis_commands_work_without_external_python(monkeypatch, command, module_name):
    calls = []
    module = types.ModuleType('research.scripts.'+module_name)
    module.main = lambda args: calls.append(args) or 0
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setenv('LIPFLOW_HOME', 'previous')
    assert cli.main(['collect', command, '--help']) == 0
    assert calls == [['--help']]
