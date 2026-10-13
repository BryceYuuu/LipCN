"""Manually collect genuine silent clips, camera only. Nothing records on --help.

SPACE starts/stops each clip; S declares no voice/whisper; ENTER confirms exact
text and name/number annotations; P replays; R retakes; ESC discards and quits.
Full-frame MP4 and a hashed JSON manifest remain in the local output directory.

Audit: python -m research.scripts.audit_silent_data --help
Errors: python -m research.scripts.report_silent_errors --help
Protocol: research/SILENT_EVALUATION.md
"""
from __future__ import annotations

import argparse
from datetime import date
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT


def main(argv=None):
    profile = Path(os.environ.get('LIPCN_PROFILE_DIR', str(Path.home() / 'Library/Application Support/LipCN')))
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, default=profile / 'silent-data', help='dataset root; contains separate train/dev/test directories')
    parser.add_argument('--speaker', required=True, help='stable pseudonymous person ID; one split per person')
    parser.add_argument('--session', required=True, help='actual session ID; a new ID on each recording date')
    parser.add_argument('--split', required=True, choices=['train', 'dev', 'test'])
    parser.add_argument('--date', default=date.today().isoformat(), help='actual local recording date YYYY-MM-DD (default: today)')
    parser.add_argument('--lighting', required=True, help='human description, e.g. daylight-front or indoor-side')
    parser.add_argument('--prompts', type=Path, help='JSON array: reference plus entities [{kind: person_name|number, text: ...}]; otherwise enter interactively')
    parser.add_argument('--camera', default='auto', help='physical camera number or name')
    parser.add_argument('--max-seconds', type=float, default=15.)
    parser.add_argument('--max-megabytes', type=float, default=512., help='maximum storage per split')
    args = parser.parse_args(argv)
    os.environ.setdefault('OPENCV_AVFOUNDATION_SKIP_AUTH', '1')
    from research.silent_collection import collect, metadata, load_prompts
    try:
        details = metadata(args.speaker, args.session, args.split, args.date, args.lighting)
        cards = load_prompts(args.prompts) if args.prompts else None
        result = collect(args.output, details, cards=cards, camera=args.camera,
                         max_seconds=args.max_seconds, max_megabytes=args.max_megabytes)
    except (OSError, ValueError, RuntimeError) as exc:
        parser.error(str(exc))
    except (KeyboardInterrupt, EOFError):
        print('\n已结束；未确认片段已丢弃。')
        return 0
    print('本地清单：' + str(result) if result.exists() else '未保存已确认片段。')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
