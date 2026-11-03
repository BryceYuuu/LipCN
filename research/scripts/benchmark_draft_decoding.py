#!/usr/bin/env python3
"""Compare fast "draft" lip decodes with the desktop's full search on labeled clips.

Uses the desktop's exact recognizer (CNVSRC base + local adapter, beam 40,
CTC 0.1, reverse 0.3, pool 10) and reports pooled CER and per-clip latency for:
CTC greedy, small attention beams (nbest 1, no reverse scoring) and the full
search. Encoding is timed once and shared, because every route reuses it.

    .venv/bin/python research/scripts/benchmark_draft_decoding.py \
        --manifest dev33.json --checkpoint ... --source-dir ... --adapter ...
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))

FULL = dict(beam_size=40, ctc_weight=.1, nbest=10, reverse_weight=.3,
            reverse_scoring='batched', candidate_pool_size=10, pre_beam_ratio=1.5)


def _summary(errors, characters, seconds):
    ordered = sorted(seconds)
    p95 = ordered[min(len(ordered) - 1, round(.95 * (len(ordered) - 1)))] if ordered else None
    return {'errors': errors, 'characters': characters,
            'cer': errors / characters if characters else None,
            'mean_seconds': sum(seconds) / len(seconds) if seconds else None, 'p95_seconds': p95}


def run(args) -> dict:
    import torch
    from evaluate_chinese import _prepared_visual_input
    from research.evaluation import load_manifest
    from evaluate_cnvsrc import _load_adapter, _reader, configured_hypotheses
    from research.evaluation import character_error

    torch.set_num_threads(4)
    reader = _reader(Path(args.checkpoint), Path(args.source_dir), FULL['beam_size'],
                     FULL['ctc_weight'], args.device)
    _load_adapter(reader, Path(args.adapter))
    reader.warmup()
    beam = reader.beam
    full_beam, full_pre_beam = beam.beam_size, beam.pre_beam_size
    routes = ['greedy'] + [f'beam{size}' for size in args.small_beams] + ['full']
    totals = {route: [0, 0, []] for route in routes}
    encode_seconds, rows = [], []
    for sample in load_manifest(args.manifest).samples:
        rois, _, _, _ = _prepared_visual_input(sample, reader, args)
        started = time.monotonic()
        encoded = reader.encode(rois)
        encode_seconds.append(time.monotonic() - started)
        outputs = {}
        started = time.monotonic()
        outputs['greedy'] = (reader.greedy(encoded), time.monotonic() - started)
        for size in args.small_beams:
            beam.beam_size, beam.pre_beam_size = size, int(1.5 * size)
            try:
                started = time.monotonic()
                text = reader.hypotheses(encoded, nbest=1)[0].text
                outputs[f'beam{size}'] = (text, time.monotonic() - started)
            finally:
                beam.beam_size, beam.pre_beam_size = full_beam, full_pre_beam
        started = time.monotonic()
        hypotheses, _ = configured_hypotheses(reader, encoded, **FULL)
        outputs['full'] = (hypotheses[0].text if hypotheses else '', time.monotonic() - started)
        row = {'id': sample.id}
        for route, (text, seconds) in outputs.items():
            errors, characters = character_error(text, sample.reference)
            totals[route][0] += errors
            totals[route][1] += characters
            totals[route][2].append(seconds)
            row[route] = {'errors': errors, 'seconds': round(seconds, 4)}
        rows.append(row)
        print(sample.id, ' '.join(f"{route}={row[route]['errors']}/{characters}" for route in routes),
              file=sys.stderr)
    return {'clips': len(rows), 'device': str(reader.enc_device),
            'encode': _summary(0, 0, encode_seconds),
            'routes': {route: _summary(*totals[route]) for route in routes}, 'per_clip': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--device', default='mps')
    parser.add_argument('--small-beams', type=lambda value: [int(v) for v in value.split(',')],
                        default=[2, 4, 8])
    parser.add_argument('--output')
    args = parser.parse_args()
    args.visual_preprocessing, args.visual_stress, args.visual_seed = 'identity', 'clean', 0
    report = run(args)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        Path(args.output).write_text(text + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('clips', 'device', 'encode', 'routes')},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
