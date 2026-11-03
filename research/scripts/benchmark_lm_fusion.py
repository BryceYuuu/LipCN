#!/usr/bin/env python3
"""Measure CMLR character-LM fusion against the desktop's Mandarin search.

Every route uses the desktop recognizer (CNVSRC base + local adapter, beam 40,
CTC 0.1, reverse 0.3, pool 10) and shares one encoding per clip:

* ``base``: the shipped search, unchanged.
* ``rescore:L:B``: the shipped candidates re-ranked by score + L*LM + B*length.
* ``fusion:L:B``: the LM joins beam search with weight L and length bonus B,
  followed by the same reverse re-ranking.

Choose settings on a development manifest, then confirm once on a held-out one.
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


def _pairs(value):
    return [tuple(float(x) for x in item.split(':')) for item in value.split(',') if item]


def _summary(errors, characters, seconds):
    ordered = sorted(seconds)
    return {'errors': errors, 'characters': characters, 'cer': errors / characters,
            'mean_seconds': sum(seconds) / len(seconds),
            'p95_seconds': ordered[min(len(ordered) - 1, round(.95 * (len(ordered) - 1)))]}


def run(args) -> dict:
    import espnet.nets.batch_beam_search as batch_module
    import torch
    from evaluate_chinese import _prepared_visual_input
    from evaluate_cnvsrc import _load_adapter, _reader, configured_hypotheses
    from research.evaluation import character_error, load_manifest
    from research.lm_fusion import MappedLM, load_cmlr_lm

    torch.set_num_threads(4)
    reader = _reader(Path(args.checkpoint), Path(args.source_dir), FULL['beam_size'],
                     FULL['ctc_weight'], args.device)
    _load_adapter(reader, Path(args.adapter))
    reader.warmup()
    lm, vocabulary = load_cmlr_lm(Path(args.lm))
    mapped = MappedLM(lm, vocabulary, reader.token_list)
    eos = len(reader.token_list) - 1
    original_beam = batch_module.BatchBeamSearch
    fusion = {}

    class FusedBeamSearch(original_beam):
        def __init__(self, *, weights, scorers, **kwargs):
            weights = {**weights, 'lm': fusion['lm'], 'length_bonus': fusion['length_bonus']}
            super().__init__(weights=weights, scorers={**scorers, 'lm': mapped}, **kwargs)

    routes = (['base'] + [f'rescore:{l}:{b}' for l, b in args.rescore]
              + [f'fusion:{l}:{b}' for l, b in args.fusion])
    totals = {route: [0, 0, []] for route in routes}
    rows = []
    for sample in load_manifest(args.manifest).samples:
        rois, _, _, _ = _prepared_visual_input(sample, reader, args)
        encoded = reader.encode(rois)
        outputs = {}
        started = time.monotonic()
        hypotheses, evidence = configured_hypotheses(reader, encoded, **FULL)
        base_seconds = time.monotonic() - started
        outputs['base'] = (hypotheses[0].text if hypotheses else '', base_seconds)
        started = time.monotonic()
        lm_scores = mapped.sequence_log_probability([list(row['token_ids']) for row in evidence], eos)
        lm_seconds = time.monotonic() - started
        for weight, bonus in args.rescore:
            ranked = sorted(zip(evidence, lm_scores), reverse=True,
                            key=lambda pair: pair[0]['score'] + weight * pair[1]
                            + bonus * len(pair[0]['token_ids']))
            outputs[f'rescore:{weight}:{bonus}'] = (ranked[0][0]['text'] if ranked else '',
                                                    base_seconds + lm_seconds)
        batch_module.BatchBeamSearch = FusedBeamSearch
        try:
            for weight, bonus in args.fusion:
                fusion.update(lm=weight, length_bonus=bonus)
                started = time.monotonic()
                fused, _ = configured_hypotheses(reader, encoded, **FULL)
                outputs[f'fusion:{weight}:{bonus}'] = (fused[0].text if fused else '',
                                                       time.monotonic() - started)
        finally:
            batch_module.BatchBeamSearch = original_beam
        row = {'id': sample.id}
        for route in routes:
            text, seconds = outputs[route]
            errors, characters = character_error(text, sample.reference)
            totals[route][0] += errors
            totals[route][1] += characters
            totals[route][2].append(seconds)
            row[route] = {'errors': errors, 'text': text, 'seconds': round(seconds, 3)}
        row['reference_characters'] = characters
        rows.append(row)
        print(sample.id, ' '.join(f'{route}={row[route]["errors"]}' for route in routes),
              f'/{characters}', file=sys.stderr, flush=True)
    return {'clips': len(rows), 'lm_vocabulary_coverage': mapped.coverage,
            'routes': {route: _summary(*totals[route]) for route in routes}, 'per_clip': rows}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--source-dir', required=True)
    parser.add_argument('--adapter', required=True)
    parser.add_argument('--lm', required=True, help='directory with CMLR lm/model.pth and model.json')
    parser.add_argument('--device', default='mps')
    parser.add_argument('--rescore', type=_pairs, default=[], help='LM:BONUS pairs, e.g. 0.2:0,0.3:0.5')
    parser.add_argument('--fusion', type=_pairs, default=[], help='LM:BONUS pairs for in-search fusion')
    parser.add_argument('--output')
    args = parser.parse_args()
    args.visual_preprocessing, args.visual_stress, args.visual_seed = 'identity', 'clean', 0
    report = run(args)
    if args.output:
        Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n',
                                     encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('clips', 'lm_vocabulary_coverage', 'routes')},
                     ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
