"""Read-only diagnostics and explicit, verified local model preparation.

No native backend imports, device access or automatic downloads. GUI callers may
use check() on the UI thread and prepare(download=True) on their setup worker.
"""
from __future__ import annotations

import hashlib
import importlib.metadata
import os
from pathlib import Path
import platform
import re
import urllib.error
import urllib.request

from .assets import Asset, plan


DEPENDENCIES = ('numpy', 'torch', 'mediapipe', 'opencv-contrib-python', 'mlx-lm',
                'pyobjc-framework-Cocoa', 'pyobjc-framework-AVFoundation',
                'pyobjc-framework-Quartz', 'faster-whisper', 'ctranslate2',
                'onnxruntime', 'opencc-python-reimplemented', 'scipy', 'sounddevice')


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def check(mode='speech', *, verify=False, environment=None) -> dict:
    """Inspect platform, package metadata and assets; never import/load a model."""
    checks = []
    for asset in plan(mode, environment=environment):
        state = 'missing'
        if asset.path.is_file():
            size = asset.path.stat().st_size
            state = 'ready' if size > 0 and (asset.size is None or size == asset.size) else 'invalid_size'
            if state == 'ready' and verify and asset.sha256 and sha256(asset.path) != asset.sha256:
                state = 'invalid_hash'
        checks.append({'key': asset.key, 'group': asset.group, 'path': str(asset.path),
                       'status': state, 'size': asset.size, 'sha256': asset.sha256,
                       'verified': bool(verify and asset.sha256 and state == 'ready'),
                       'downloadable': asset.url is not None, 'source': asset.source,
                       'research_license': asset.research_license})
    dependencies = []
    for name in DEPENDENCIES:
        try:
            version = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            version = None
        dependencies.append({'name': name, 'version': version, 'status': 'ready' if version else 'missing'})
    system = {'system': platform.system(), 'machine': platform.machine(),
              'python': platform.python_version(), 'macos': platform.mac_ver()[0]}
    supported = system['system'] == 'Darwin' and system['machine'] == 'arm64'
    supported = supported and platform.python_version_tuple()[:2] in [('3', '11'), ('3', '12')]
    if supported and system['macos']:
        supported = int(system['macos'].split('.')[0]) >= 14
    return {'schema_version': 1, 'mode': mode, 'platform': system, 'platform_supported': supported,
            'ready': supported and all(c['status'] == 'ready' for c in checks + dependencies),
            'assets': checks, 'dependencies': dependencies,
            'missing_download_bytes': sum(c['size'] or 0 for c in checks if c['status'] != 'ready' and c['downloadable']),
            'note': 'Read-only check; devices, model compatibility and permissions were not exercised.'}


def download(asset: Asset, *, progress=None, opener=urllib.request.urlopen) -> None:
    """Resume pinned bytes when supported; publish only after size and SHA256 match."""
    if not asset.url or not asset.sha256 or asset.size is None:
        raise ValueError('This asset must be supplied locally: ' + asset.key)
    if asset.path.is_file() and asset.path.stat().st_size == asset.size and sha256(asset.path) == asset.sha256:
        if progress:
            progress({'key': asset.key, 'state': 'verified', 'received': asset.size, 'total': asset.size})
        return
    asset.path.parent.mkdir(parents=True, exist_ok=True)
    partial = asset.path.with_name(asset.path.name + '.lipcn-part')
    offset = partial.stat().st_size if partial.is_file() else 0
    if offset > asset.size:
        partial.unlink()
        offset = 0
    if offset < asset.size:
        headers = {'User-Agent': 'LipCN/2.3.0', 'Accept-Encoding': 'identity'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        request = urllib.request.Request(asset.url, headers=headers)
        with opener(request, timeout=60) as response:
            status = getattr(response, 'status', 200)
            if status == 206:
                content_range = response.headers.get('Content-Range', '')
                match = re.fullmatch(r'bytes (\d+)-(\d+)/(\d+)', content_range)
                if not match or int(match[1]) != offset or int(match[3]) != asset.size:
                    raise ValueError('Invalid resume response for ' + asset.key)
            elif status == 200:
                offset = 0  # Server ignored Range: restart rather than append duplicate bytes.
            else:
                raise ValueError(f'Unexpected HTTP {status} for {asset.key}')
            with partial.open('ab' if offset else 'wb') as output:
                received = offset
                while chunk := response.read(1024 * 1024):
                    received += len(chunk)
                    if received > asset.size:
                        raise ValueError('Download exceeds pinned size for ' + asset.key)
                    output.write(chunk)
                    if progress:
                        progress({'key': asset.key, 'state': 'downloading', 'received': received, 'total': asset.size})
                output.flush()
                os.fsync(output.fileno())
    if partial.stat().st_size != asset.size:
        raise ValueError('Incomplete download; run setup again to resume: ' + asset.key)
    if sha256(partial) != asset.sha256:
        partial.unlink()  # A corrupt completed partial must not poison the next attempt.
        raise ValueError('SHA256 mismatch; previous installed file was preserved: ' + asset.key)
    os.replace(partial, asset.path)
    if progress:
        progress({'key': asset.key, 'state': 'verified', 'received': asset.size, 'total': asset.size})


def prepare(mode='speech', *, download=False, accept_research_license=False,
            progress=None, environment=None) -> dict:
    """No mutation unless download=True. Local/private adapters are never downloaded."""
    assets = plan(mode, environment=environment)
    if not download:
        return check(mode, environment=environment)
    if any(asset.research_license for asset in assets) and not accept_research_license:
        raise ValueError('Visual research assets require explicit acceptance of their source terms; '
                         'use accept_research_license=True (CLI: --accept-research-license) or choose speech mode.')
    downloader = globals()['download']
    for asset in assets:
        if asset.url is not None:
            downloader(asset, progress=progress)
    return check(mode, verify=True, environment=environment)
