"""Current LipCN entry; the separate `lipflow` command retains the legacy CLI."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import platform
import runpy
import sys

from . import __version__
from .assets import MODES, paths


def desktop_path() -> Path:
    return Path(__file__).resolve().parents[1] / 'desktop/mandarin/live_test.py'


def launch() -> int:
    if platform.system() != 'Darwin' or platform.machine() != 'arm64':
        raise ValueError('LipCN desktop requires macOS 14+ on Apple Silicon. `lipcn doctor` is available without devices.')
    entry = desktop_path()
    if not entry.is_file():
        raise ValueError('Desktop source is missing; reinstall LipCN with its desktop package.')
    for name in ('HF_HUB_OFFLINE', 'TRANSFORMERS_OFFLINE', 'HF_HUB_DISABLE_TELEMETRY', 'DO_NOT_TRACK'):
        os.environ[name] = '1'
    os.environ['TOKENIZERS_PARALLELISM'] = 'false'
    os.environ.setdefault('LIPCN_FACE_MODEL', str(paths()['FACE_MODEL']))
    os.environ['LIPFLOW_HOME'] = os.environ.get('LIPCN_PROFILE_DIR', str(Path.home() / 'Library/Application Support/LipCN'))
    sys.path.insert(0, str(entry.parent))
    sys.argv = [str(entry)]
    runpy.run_path(str(entry), run_name='__main__')
    return 0


def show_report(report, *, as_json=False):
    if as_json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        return
    print('LipCN ' + __version__ + ' · ' + report['mode'])
    print('Platform: ' + ('ready' if report['platform_supported'] else 'unsupported'))
    for item in report['dependencies']:
        print(f"{item['status']:12} {item['name']} {item['version'] or ''}")
    for item in report['assets']:
        note = ' (provide your compatible local adapter)' if not item['downloadable'] else ''
        print(f"{item['status']:12} {item['key']}: {item['path']}{note}")
    print(report['note'])


def main(argv=None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv and argv[0] == 'collect':
        os.environ['LIPFLOW_HOME'] = os.environ.get(
            'LIPCN_PROFILE_DIR', str(Path.home() / 'Library/Application Support/LipCN'))
        if len(argv) > 1 and argv[1] == 'audit':
            from research.scripts.audit_silent_data import main as collect
            return collect(argv[2:]) or 0
        if len(argv) > 1 and argv[1] == 'report':
            from research.scripts.report_silent_errors import main as collect
            return collect(argv[2:]) or 0
        from research.scripts.collect_silent_chinese import main as collect
        return collect(argv[1:]) or 0
    parser = argparse.ArgumentParser(prog='lipcn', description='LipCN three-language desktop and offline setup tools.')
    parser.add_argument('--version', action='version', version='LipCN ' + __version__)
    commands = parser.add_subparsers(dest='command')
    commands.add_parser('run', help='open the current desktop (default: Simplified Chinese)')
    commands.add_parser('collect', help='explicitly collect or audit local silent-video research data')
    doctor = commands.add_parser('doctor', help='inspect dependencies/assets without models, devices or network')
    doctor.add_argument('--mode', choices=MODES, default='speech')
    doctor.add_argument('--verify', action='store_true', help='also hash local model files (can take time)')
    doctor.add_argument('--json', action='store_true')
    setup = commands.add_parser('setup', help='plan model preparation; downloads require --download')
    setup.add_argument('--mode', choices=MODES)
    setup.add_argument('--download', action='store_true')
    setup.add_argument('--accept-research-license', action='store_true')
    setup.add_argument('--json', action='store_true')
    args = parser.parse_args(argv)
    try:
        if args.command in (None, 'run'):
            return launch()
        from .setup import check, prepare
        if args.command == 'doctor':
            report = check(args.mode, verify=args.verify)
        else:
            mode = args.mode
            if mode is None:
                if not sys.stdin.isatty():
                    parser.error('setup needs --mode in a noninteractive session')
                print('Choose mode: speech / mandarin / english / all')
                mode = input('Mode [speech]: ').strip() or 'speech'
                if mode not in MODES:
                    parser.error('unknown setup mode')
            if args.download:
                print('Downloading only explicitly selected public assets; private adapters are never downloaded.', file=sys.stderr)
            report = prepare(mode, download=args.download,
                             accept_research_license=args.accept_research_license)
        show_report(report, as_json=args.json)
        return 0 if report['ready'] else 1
    except (ValueError, OSError) as exc:
        print('LipCN: ' + str(exc), file=sys.stderr)
        return 2
