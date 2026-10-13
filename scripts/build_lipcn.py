"""Build a relocatable Apple Silicon bundle with CPython and its dependencies.

Prepare a pinned environment with `uv sync --frozen --extra desktop --no-editable`.
Then: python scripts/build_lipcn.py --python /path/to/venv/bin/python --output /path/to/dist
Models, private adapters, profiles and developer checkout paths are never copied.
The result is ad-hoc signed; Developer ID signing/notarization is a separate release step.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import platform
import plistlib
import shutil
import subprocess
import sys
import tempfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]


def run(*args, **kwargs):
    return subprocess.run(list(map(str, args)), check=True, **kwargs)


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            result.update(chunk)
    return result.hexdigest()


def ignored(folder, names):
    return [name for name in names if name in {'__pycache__', '.pytest_cache', '.git', '.DS_Store',
                                               'direct_url.json', '_virtualenv.pth', '_virtualenv.py'}
            or name.startswith('_editable') or name.endswith(('.pyc', '.pyo'))]


class SpaceGuard:
    """Never let a new archive consume the reserved free disk space."""
    def __init__(self, stream, directory, reserve=512 * 1024 * 1024):
        self.stream, self.directory, self.reserve = stream, directory, reserve

    def __getattr__(self, name):
        return getattr(self.stream, name)

    def write(self, data):
        if shutil.disk_usage(self.directory).free - len(data) < self.reserve:
            raise OSError('ZIP stopped to preserve 512 MiB free space; the verified app is retained.')
        return self.stream.write(data)


def archive_bundle(app, destination):
    """Stream files into an atomic ZIP, retaining executable modes and empty dirs."""
    partial = destination.with_name(destination.name + '.part')
    try:
        with partial.open('w+b') as stream:
            with zipfile.ZipFile(SpaceGuard(stream, destination.parent), 'w',
                                 compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
                for path in [app, *sorted(app.rglob('*'))]:
                    if path.is_symlink():
                        raise RuntimeError('The standalone bundle must not contain external symlinks.')
                    info = zipfile.ZipInfo.from_file(path, str(path.relative_to(app.parent)))
                    info.compress_type = zipfile.ZIP_DEFLATED
                    if path.is_dir():
                        archive.writestr(info, b'')
                    else:
                        with path.open('rb') as source, archive.open(info, 'w', force_zip64=True) as target:
                            for chunk in iter(lambda: source.read(1024 * 1024), b''):
                                target.write(chunk)
        os.replace(partial, destination)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise


def environment(python):
    code = """import json,sys,sysconfig,importlib.metadata
print(json.dumps({'base':sys.base_prefix,'version':list(sys.version_info[:3]),'site':sysconfig.get_path('purelib'),
'packages':{d.metadata['Name']:d.version for d in importlib.metadata.distributions()}}))"""
    return json.loads(subprocess.check_output([str(python), '-c', code], text=True))


def build(python: Path | None, output: Path, *, archive=True, reuse_runtime=False, reuse_native_check=False):
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise RuntimeError('Build on an Apple Silicon Mac.')
    output = output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    environment_file = output/'.build-environment.json'
    if python is None:
        if not reuse_runtime or not environment_file.is_file():
            raise RuntimeError('Provide --python on the first build; reuse requires saved build environment metadata.')
        saved = json.loads(environment_file.read_text())
        if saved['lock_sha256'] != digest(ROOT/'uv.lock'):
            raise RuntimeError('Dependency lock changed; build a fresh runtime.')
        info = saved['environment']
    else:
        info = environment(python)
        environment_file.write_text(json.dumps({'lock_sha256':digest(ROOT/'uv.lock'),'environment':info},indent=2)+'\n')
    if info['version'][:2] != [3, 12]:
        raise RuntimeError('The reproducible bundle launcher requires CPython 3.12.')
    base = Path(info['base'])
    if not (base/'lib/libpython3.12.dylib').is_file():
        raise RuntimeError('Use a relocatable python-build-standalone runtime (for example uv Python).')
    version = tomllib.loads((ROOT/'pyproject.toml').read_text())['project']['version']
    app = output/'LipCN.app'
    previous_probe = None
    if reuse_native_check:
        if not reuse_runtime:
            raise RuntimeError('--reuse-native-check requires --reuse-runtime.')
        previous = json.loads((app/'Contents/Resources/BUILD-MANIFEST.json').read_text())
        if previous['lock_sha256'] != digest(ROOT/'uv.lock') or previous['native_import_probe']['imports'] != 'passed':
            raise RuntimeError('No matching successful native-import validation to reuse.')
        run('codesign','--verify','--deep','--strict',app,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
        previous_probe = previous['native_import_probe']
    if app.exists() and not reuse_runtime:
        raise RuntimeError('Output bundle already exists; use a fresh output directory.')
    contents = app/'Contents'
    resources, macos = contents/'Resources', contents/'MacOS'
    resources.mkdir(parents=True, exist_ok=reuse_runtime)
    macos.mkdir(exist_ok=reuse_runtime)
    runtime = resources/'runtime'
    if not reuse_runtime:
        print('Copying the standalone Python runtime and locked dependencies…', file=sys.stderr, flush=True)
        shutil.copytree(base, runtime, symlinks=False, ignore=ignored)
    elif not (runtime/'lib/libpython3.12.dylib').is_file():
        raise RuntimeError('--reuse-runtime requires a previously staged LipCN runtime.')
    target_site = runtime/'lib/python3.12/site-packages'
    target_site.mkdir(parents=True, exist_ok=True)
    if not reuse_runtime:
        shutil.copytree(info['site'], target_site, dirs_exist_ok=True, symlinks=False, ignore=ignored)
    # App code is staged separately and wins over any installed wheel snapshot.
    for item in tuple(target_site.iterdir()):
        if item.name in {'lipcn', 'lipflow', 'espnet', 'desktop', 'research'} or item.name.startswith(('lipcn-', 'lipflow-')):
            if item.is_dir(): shutil.rmtree(item)
            else: item.unlink()
    source = resources/'source'
    if source.exists():
        shutil.rmtree(source)
    source.mkdir()
    # A freshly built wheel defines the shipped source/data inventory.
    with tempfile.TemporaryDirectory(prefix='lipcn-wheel-') as temporary:
        run('uv', 'build', '--wheel', '--out-dir', temporary, ROOT, stdout=subprocess.PIPE)
        wheel = next(Path(temporary).glob('*.whl'))
        with zipfile.ZipFile(wheel) as bundle:
            for name in bundle.namelist():
                relative = Path(name)
                if '..' in relative.parts or relative.is_absolute():
                    raise RuntimeError('Unsafe wheel path')
                if relative.parts[0].endswith('.dist-info'):
                    target = target_site/name
                else:
                    target = source/name
                if name.endswith('/'):
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(bundle.read(name))
    for name in ('LICENSE', 'NOTICE', 'uv.lock', 'pyproject.toml'):
        shutil.copy2(ROOT/name, source/name)
    shutil.copytree(ROOT/'LICENSES', source/'LICENSES', dirs_exist_ok=True)
    shutil.copy2(ROOT/'scripts/THIRD-PARTY-NOTICE.md', resources/'THIRD-PARTY-NOTICE.md')
    # Drop source/build tools not needed to execute the standalone interpreter.
    for relative in ('include', 'share/man', 'lib/pkgconfig'):
        shutil.rmtree(runtime/relative, ignore_errors=True)
    lib = runtime/'lib/libpython3.12.dylib'
    run('install_name_tool', '-id', '@rpath/libpython3.12.dylib', lib)
    launcher = macos/'LipCN'
    run('clang', '-O2', '-arch', 'arm64', '-mmacosx-version-min=14.0',
        '-I'+str(base/'include/python3.12'), ROOT/'scripts/lipcn_launcher.c',
        '-L'+str(runtime/'lib'), '-lpython3.12',
        '-Wl,-rpath,@executable_path/../Resources/runtime/lib', '-o', launcher)
    plist = {'CFBundleName':'LipCN','CFBundleDisplayName':'LipCN','CFBundleIdentifier':'app.lipcn.desktop',
             'CFBundleVersion':'8','CFBundleShortVersionString':version,'CFBundlePackageType':'APPL',
             'CFBundleExecutable':'LipCN','LSMinimumSystemVersion':'14.0','NSHighResolutionCapable':True,
             'NSCameraUsageDescription':'LipCN reads mouth movement locally after you start a sentence. Video is not saved or uploaded.',
             'NSMicrophoneUsageDescription':'LipCN recognizes Mandarin speech locally after you start a sentence. Audio is not saved or uploaded.'}
    with (contents/'Info.plist').open('wb') as stream:
        plistlib.dump(plist, stream)
    # Refuse a bundle that still dynamically links to a developer's runtime/Homebrew.
    binaries = [launcher, lib, runtime/'bin/python3.12'] + list(runtime.rglob('*.so')) + list(runtime.rglob('*.dylib'))
    native_files = sorted(set(binaries))
    print(f'Auditing and signing {len(native_files)} native libraries…', file=sys.stderr, flush=True)
    for binary in native_files:
        if binary.suffix == '.dylib':
            # Wheel build IDs can contain the upstream build machine's absolute path.
            # References still undergo the strict external-link check below.
            run('install_name_tool', '-id', '@rpath/'+binary.name, binary,
                stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        result = subprocess.run(['otool','-L',str(binary)], text=True,capture_output=True)
        if result.returncode:
            continue
        for line in result.stdout.splitlines()[1:]:
            if not line.startswith((' ', '\t')):  # Per-architecture headers in universal binaries.
                continue
            dependency = line.strip().split(' (',1)[0]
            if dependency.startswith('/') and not dependency.startswith(('/usr/lib/', '/System/Library/')):
                raise RuntimeError(f'External native dependency in {binary.relative_to(app)}: {dependency}')
    for binary in native_files:
        run('codesign', '--force', '--sign', '-', binary, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    run('codesign', '--force', '--sign', '-', '--identifier', plist['CFBundleIdentifier'], app,
        stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    # Test after moving: neither the build output path nor the original venv may supply imports.
    relocated = output/'relocation-check'/'LipCN.app'
    relocated.parent.mkdir()
    app.rename(relocated)
    clean_env = {key:value for key,value in os.environ.items() if not key.startswith(('PYTHON', 'LIPCN_', 'LIPFLOW_'))}
    clean_env['PYTHONNOUSERSITE'] = '1'
    try:
        print('Verifying the relocated launcher, offline doctor and native imports…', file=sys.stderr, flush=True)
        actual = subprocess.check_output([str(relocated/'Contents/MacOS/LipCN'),'--version'],cwd='/tmp',env=clean_env,text=True)
        if actual.strip() != 'LipCN '+version:
            raise RuntimeError('Relocated launcher version mismatch')
        doctor = subprocess.run([str(relocated/'Contents/MacOS/LipCN'),'doctor','--mode','all','--json'],
                                cwd='/tmp',env=clean_env,text=True,capture_output=True)
        report = json.loads(doctor.stdout)
        if doctor.returncode not in (0,1) or any(d['status']!='ready' for d in report['dependencies']):
            raise RuntimeError('Bundled dependency metadata is incomplete: '+doctor.stdout+doctor.stderr)
        # Import native modules only. No model construction, tensor allocation or device capture.
        probe = '''import sys, pathlib, json
import cv2, mediapipe, torch, sounddevice, objc, AppKit, AVFoundation, Quartz
import faster_whisper, ctranslate2, onnxruntime, opencc, scipy, mlx.core, mlx_lm
root=pathlib.Path(sys.prefix).resolve()
modules=[cv2,mediapipe,torch,sounddevice,objc,AppKit,AVFoundation,Quartz,faster_whisper,ctranslate2,onnxruntime,opencc,scipy,mlx.core,mlx_lm]
assert all(pathlib.Path(m.__file__).resolve().is_relative_to(root) for m in modules), 'external module path'
print(json.dumps({'imports':'passed','native_module_count':len(modules)}))'''
        if previous_probe is None:
            probe_result = subprocess.check_output([str(relocated/'Contents/Resources/runtime/bin/python3.12'),'-I','-B','-c',probe],
                                                   cwd='/tmp',env=clean_env,text=True)
        else:
            probe_result = json.dumps({**previous_probe, 'reused_after_signature_verification':True})
    finally:
        relocated.rename(app)
        relocated.parent.rmdir()
    manifest = {'version':version,'architecture':'arm64','python_version':info['version'],
                'lock_sha256':digest(ROOT/'uv.lock'),'packages':info['packages'],
                'native_import_probe':json.loads(probe_result),'relocation':'passed',
                'models_included':False,'signing':'ad-hoc; not Apple notarized',
                'note':'Runtime/import and offline-doctor validation only; no GUI, device or model inference.'}
    (resources/'BUILD-MANIFEST.json').write_text(json.dumps(manifest,indent=2)+'\n')
    # Final manifest changes resources: sign the enclosing bundle again.
    run('codesign','--force','--sign','-','--identifier',plist['CFBundleIdentifier'],app,
        stdout=subprocess.DEVNULL,stderr=subprocess.PIPE)
    run('codesign','--verify','--deep','--strict',app,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    archive_path = output/f'LipCN-{version}-macOS-arm64.zip'
    if archive:
        print('Creating the distributable ZIP…', file=sys.stderr, flush=True)
        archive_bundle(app, archive_path)
        archive_path.with_suffix('.zip.sha256').write_text(digest(archive_path)+'  '+archive_path.name+'\n')
    print(json.dumps({'app':str(app),'archive':str(archive_path) if archive else None,'manifest':manifest},indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--python',type=Path,help='locked build environment interpreter; required only for the first build')
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--no-archive',action='store_true')
    parser.add_argument('--reuse-runtime',action='store_true',help='refresh code and revalidate an existing staged runtime without copying it again')
    parser.add_argument('--reuse-native-check',action='store_true',help='reuse a successful native-import check only after verifying the unchanged signed bundle and dependency lock')
    arguments = parser.parse_args()
    build(arguments.python,arguments.output,archive=not arguments.no_archive,reuse_runtime=arguments.reuse_runtime,
          reuse_native_check=arguments.reuse_native_check)
