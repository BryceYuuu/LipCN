"""Compare current CMLR, original CNVSRC and a frozen CNVSRC candidate locally.

Every arm receives the SAME source pixels. Preparation is shared in a bounded
RAM cache and its measured cost is charged to each arm's latency. No audio, LLM
cleanup, reference-guided decoding, or test-time parameter selection occurs.
Outputs contain raw local predictions; do not distribute source video/weights.
"""
from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from _lipflow_research_path import REPOSITORY_ROOT

import argparse
from contextlib import contextmanager
from dataclasses import replace
import hashlib
import json
import math
import re
import time
from types import SimpleNamespace

from research.evaluation import Prediction, Thresholds, evaluate, load_manifest


def _hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def selected_preset(path):
    if path is None:
        return 'identity', None
    data = json.loads(Path(path).read_text(encoding='utf-8'))
    if not isinstance(data, dict):
        raise ValueError('selection must contain a frozen dev-only preprocessing choice')
    preset = data.get('selected_preprocessing')
    if (type(data.get('schema_version')) is not int or data.get('schema_version') != 1 or data.get('selection_partition') != 'dev'
            or data.get('test_used_for_selection') is not False
            or preset not in ('identity', 'stabilize_mild', 'stabilize_moderate')):
        raise ValueError('selection must contain a frozen dev-only preprocessing choice')
    provenance = data.get('provenance', {})
    if not isinstance(provenance, dict):
        raise ValueError('selection requires hashes of its development reports')
    hashes = provenance.get('dev_report_sha256') or data.get('input_sha256')
    if (not isinstance(hashes, dict) or not hashes or any(not isinstance(k, str) or not isinstance(v, str)
            or not re.fullmatch('[0-9a-f]{64}', v) for k, v in hashes.items())):
        raise ValueError('selection requires hashes of its development reports')
    return preset, data


def prepare_shared(dataset, loader, reader, *, max_bytes):
    """Only images and identities enter the cache; never retain reference text."""
    if type(max_bytes) is not int or max_bytes < 1:
        raise ValueError('positive image-cache byte budget required')
    cache, costs, used = {}, {}, 0
    for sample in dataset.samples:
        started = time.monotonic()
        try:
            images, duration, quality = loader(replace(sample, reference=''), reader)
            if images.nbytes + used > max_bytes:
                raise ValueError('shared visual cache budget exceeded; sample retained as a failure in every arm')
            images.setflags(write=False)
            cache[sample.id] = (images, duration, quality)
            used += images.nbytes
        except Exception as exc:
            cache[sample.id] = exc
        costs[sample.id] = time.monotonic() - started
        print(f'PREPARE {len(cache)}/{len(dataset.samples)} {sample.id}', file=sys.stderr, flush=True)
    return cache, costs, used


@contextmanager
def shared_loader(module, cache):
    original = module._visual_input
    def cached(sample, reader):
        if sample.reference:
            raise ValueError('reference leaked into the shared pixel boundary')
        item = cache[sample.id]
        if isinstance(item, Exception):
            raise item
        return item
    module._visual_input = cached
    try:
        yield
    finally:
        module._visual_input = original


def charge_preparation(report, dataset, costs):
    """Recompute latency gates after charging each arm its full preparation cost."""
    identities = {s.id: s for s in dataset.samples}
    rows = report['samples']
    if len(rows) != len(identities) or {r['sample_id'] for r in rows} != set(identities):
        raise ValueError('benchmark report changed the frozen sample set')
    for row in rows:
        sample = identities[row['sample_id']]
        for field, expected in (('reference', sample.reference), ('speaker', sample.speaker),
                                ('session', sample.session), ('video_sha256', sample.sha256)):
            if row.get(field) != expected:
                raise ValueError('benchmark report changed a frozen identity/reference')
    predictions = []
    for row in report['samples']:
        elapsed = row.get('processing_seconds')
        if elapsed is not None:
            elapsed += costs[row['sample_id']]
        predictions.append(Prediction(row['sample_id'], row['raw'], row.get('duration_seconds'), elapsed,
                                      row['action'], row.get('reason', ''), row.get('error'), row.get('margin')))
    scoring = evaluate(dataset, predictions)
    recomputed = {row['sample_id']: row for row in scoring['samples']}
    for old in report['samples']:
        new = recomputed[old['sample_id']]
        old['processing_seconds'] = new['processing_seconds']
        old['rtf'] = new['rtf']
        old['shared_preparation_seconds_charged'] = costs[old['sample_id']]
    report['metrics'], report['readiness'] = scoring['metrics'], scoring['readiness']
    if report['model']['visual_configuration']['stress'] != 'clean':
        report['readiness']['ready'] = False
        report['readiness']['status'] = 'not_ready'
        report['readiness']['gates'].append({'criterion': 'real_camera_evidence', 'passed': False,
                                             'observed': 'synthetic degradation'})
    return report


def verify_frozen_inputs(plan, args):
    files = {args.manifest: plan['manifest_sha256'], args.checkpoint: plan['cnvsrc_checkpoint_sha256']}
    for option, field in (('selection', 'selection_sha256'), ('adapter', 'adapter_sha256'),
                          ('reverse_validation', 'reverse_validation_sha256')):
        if getattr(args, option, None):
            files[getattr(args, option)] = plan[field]
    files.update({str(Path(args.cmlr_dir) / name): digest for name, digest in plan['cmlr_file_sha256'].items()})
    files.update({str(REPOSITORY_ROOT / name): digest for name, digest in plan['source_sha256'].items()})
    files.update(plan.get('development_report_sha256', {}))
    files.update(plan.get('binary_asset_sha256', {}))
    for path, digest in files.items():
        if _hash(path) != digest:
            raise ValueError(f'frozen benchmark input changed: {path}')


def selection_reports(path, selection, dataset):
    """Bind the local dev evidence; reject shared test identities or source files."""
    if selection is None:
        return {}
    hashes = selection.get('provenance', {}).get('dev_report_sha256')
    named = {f'dev-{name}.json': digest for name, digest in hashes.items()} if hashes else selection['input_sha256']
    frozen = {}
    test_ids = {s.id for s in dataset.samples}
    test_hashes = {s.sha256 for s in dataset.samples if s.sha256}
    for name, digest in named.items():
        report_path = (Path(path).parent / name).resolve()
        if _hash(report_path) != digest:
            raise ValueError('development report changed after selection')
        report = json.loads(report_path.read_text(encoding='utf-8'))
        if not isinstance(report, dict):
            raise ValueError('selection evidence must be a development-only report')
        rows = report.get('samples', [])
        if (not isinstance(rows, list) or not rows or any(not isinstance(r, dict)
                or not isinstance(r.get('sample_id'), str) or not r['sample_id']
                or not isinstance(r.get('raw'), str) or not isinstance(r.get('reference'), str)
                or not isinstance(r.get('video_sha256'), str) or not re.fullmatch('[0-9a-f]{64}', r['video_sha256'])
                for r in rows)
                or len({r['sample_id'] for r in rows}) != len(rows)
                or any(r.get('split') not in ('dev', 'val', 'validation') for r in rows)
                or any(r.get('sample_id') in test_ids or r.get('video_sha256') in test_hashes for r in rows)):
            raise ValueError('selection evidence must be development-only and disjoint from test videos')
        frozen[str(report_path)] = digest
    return frozen


def validate_reverse_validation(validation, plan, args):
    if not isinstance(validation, dict) or not isinstance(validation.get('source_sha256'), dict):
        raise ValueError('batched validation must contain a source-bound development proof')
    checks = {'checkpoint_sha256': plan['cnvsrc_checkpoint_sha256'], 'adapter_sha256': plan['adapter_sha256'],
              'beam_size': 40, 'ctc_weight': args.ctc_weight, 'reverse_weight': args.reverse_weight,
              'nbest': 10, 'tie_tolerance': plan['arms']['cnvsrc_candidate']['reverse_tie_tolerance']}
    rows = validation.get('rows')
    if (type(validation.get('schema_version')) is not int or validation.get('schema_version') != 1
            or validation.get('partition') != 'dev' or validation.get('passed') is not True
            or validation.get('test_used') is not False or type(validation.get('samples')) is not int
            or not isinstance(rows, list) or not rows or len(rows) != validation['samples']
            or any(validation.get(k) != v for k, v in checks.items())
            or any(validation.get('source_sha256', {}).get(k) != plan['source_sha256'][k] for k in
                   ('research/reverse_batch.py', 'research/scripts/evaluate_cnvsrc.py', 'research/scripts/compare_chinese_decoders.py'))):
        raise ValueError('batched validation does not match the frozen candidate')
    for row in rows:
        if not isinstance(row, dict):
            raise ValueError('batched development parity row is invalid')
        difference = row.get('max_absolute_reverse_difference')
        if (not isinstance(row.get('sample_id'), str) or not row['sample_id']
                or any(row.get(k) is not True for k in ('ranking_identical', 'raw_identical', 'action_identical'))
                or type(difference) not in (int, float) or not math.isfinite(difference) or not 0 <= difference < 1e-3):
            raise ValueError('batched development parity row is invalid')
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('batched development samples must be unique')


def run(args):
    import torch
    import evaluate_chinese
    import evaluate_cnvsrc
    from lipflow.vsr import LipReader

    torch.set_num_threads(args.cpu_threads)
    dataset = load_manifest(args.manifest)
    if dataset.split != 'test' or not dataset.samples or any(s.split != 'test' for s in dataset.samples):
        raise ValueError('benchmark requires a complete, frozen test manifest')
    preset, selection = selected_preset(args.selection)
    output = Path(args.output_dir)
    conditions = ['clean', 'synthetic_mild'] if args.synthetic_stress else ['clean']
    names = ('cmlr', 'cnvsrc_base', 'cnvsrc_candidate')
    output.mkdir(parents=True, exist_ok=True)
    planned_outputs = [output / 'frozen-plan.json', *[output / f'{name}-{condition}.json' for name in names for condition in conditions]]
    if any(path.exists() for path in planned_outputs):
        raise ValueError('output already contains benchmark results; choose a new output directory')
    paths = ['research/scripts/benchmark_chinese_models.py', 'research/scripts/evaluate_chinese.py',
             'research/scripts/evaluate_cnvsrc.py', 'research/scripts/compare_chinese_decoders.py',
             'research/visual_robustness.py', 'research/reverse_batch.py', 'lipflow/vsr.py', 'lipflow/face.py',
             'research/evaluation.py', 'lipflow/confidence.py', 'lipflow/chinese.py',
             'lipflow/dictation.py', 'lipflow/paths.py', 'research/scripts/_lipflow_research_path.py']
    paths.extend(str(path.relative_to(REPOSITORY_ROOT)) for path in sorted((REPOSITORY_ROOT / 'espnet').rglob('*.py')))
    plan = {'schema_version': 1, 'local_only': True, 'manifest_sha256': _hash(args.manifest),
            'selection_sha256': _hash(args.selection) if args.selection else None,
            'reverse_validation_sha256': _hash(args.reverse_validation) if args.reverse_validation else None,
            'source_sha256': {path: _hash(REPOSITORY_ROOT / path) for path in paths},
            'test_used_for_selection': False, 'candidate_preprocessing': preset,
            'decoder_selection_provenance': 'dev_model_binding_verified' if selection else 'not_verified_by_this_runner',
            'training_exposure_audit': 'Required separately by compare_chinese_cohorts; pretrained exposure remains unknown',
            'cpu_threads': args.cpu_threads, 'seed': args.seed, 'conditions': conditions,
            'cnvsrc_checkpoint_sha256': _hash(args.checkpoint),
            'adapter_sha256': _hash(args.adapter) if args.adapter else None,
            'cmlr_file_sha256': {name: _hash(Path(args.cmlr_dir) / name) for name in
                                ('vsr/model.json', 'vsr/model.pth', 'lm/model.json', 'lm/model.pth')},
            'arms': {'cmlr': {'beam_size': 10, 'decoder': 'current LipReader Mandarin recipe with its published LM'},
                     'cnvsrc_base': {'beam_size': 40, 'ctc_weight': .5, 'reverse_weight': 0.0, 'nbest': 10, 'adapter': False},
                     'cnvsrc_candidate': {'beam_size': 40, 'ctc_weight': args.ctc_weight,
                                          'reverse_weight': args.reverse_weight, 'nbest': 10, 'adapter': bool(args.adapter),
                                          'reverse_scoring': args.reverse_scoring, 'reverse_tie_tolerance': evaluate_cnvsrc.REVERSE_TIE_TOLERANCE}},
            'cache_max_bytes': args.cache_max_mib * 1024 ** 2,
            'latency': 'Shared preparation cost charged to every arm; startups separate. Shared wall time is not independent end-to-end model latency.',
            'real_webcam_claimed': False, 'models_trained_from_scratch': False}
    binary_assets = [REPOSITORY_ROOT / 'lipflow/mean_face.npy']
    if any(not sample.mouth_roi for sample in dataset.samples):
        binary_assets.append(REPOSITORY_ROOT / 'models/face_landmarker.task')
    plan['binary_asset_sha256'] = {str(p.resolve()): _hash(p) for p in binary_assets}
    if args.reverse_scoring == 'batched':
        if not args.reverse_validation:
            raise ValueError('batched layout requires a frozen real-checkpoint development validation')
        validation = json.loads(Path(args.reverse_validation).read_text(encoding='utf-8'))
        validate_reverse_validation(validation, plan, args)
    if selection:
        binding = selection['model_binding']
        checks = {'checkpoint_sha256': plan['cnvsrc_checkpoint_sha256'], 'ctc_weight': args.ctc_weight,
                  'reverse_weight': args.reverse_weight, 'beam_size': 40, 'nbest': 10}
        if any(binding.get(k) != v for k, v in checks.items()):
            raise ValueError('candidate model/decoder differs from the frozen dev choice')
        selected_adapter = binding.get('adapter')
        if (selected_adapter or {}).get('file_sha256') != plan['adapter_sha256']:
            raise ValueError('candidate adapter differs from the frozen dev choice')
    plan['development_report_sha256'] = selection_reports(args.selection, selection, dataset)
    plan_path = output / 'frozen-plan.json'
    plan_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    verify_frozen_inputs(plan, args)
    cache, costs, used = prepare_shared(dataset, evaluate_chinese._visual_input,
                                        SimpleNamespace(resample=LipReader.resample), max_bytes=plan['cache_max_bytes'])
    with shared_loader(evaluate_chinese, cache):
        for name in names:
            for condition in conditions:
                verify_frozen_inputs(plan, args)
                common = dict(manifest=args.manifest, device=args.device, visual_seed=args.seed,
                              visual_stress=condition, visual_preprocessing=preset if name == 'cnvsrc_candidate' else 'identity')
                print(f'START {name} {condition}', file=sys.stderr, flush=True)
                if name == 'cmlr':
                    limits = vars(Thresholds()).copy()
                    limits.pop('require_webcam_domain'); limits.pop('require_silent_articulation')
                    report = evaluate_chinese._batch(SimpleNamespace(**common, **limits, beam_size=10,
                        model_dir=args.cmlr_dir, allow_non_webcam_domain=False, allow_voiced_articulation=False))
                else:
                    report = evaluate_cnvsrc.run(SimpleNamespace(**common, checkpoint=args.checkpoint,
                        source_dir=args.source_dir, adapter=args.adapter if name == 'cnvsrc_candidate' else None,
                        beam_size=40, nbest=10, ctc_weight=args.ctc_weight if name == 'cnvsrc_candidate' else .5,
                        reverse_weight=args.reverse_weight if name == 'cnvsrc_candidate' else 0.0,
                        reverse_scoring=args.reverse_scoring if name == 'cnvsrc_candidate' else 'sequential'))
                charge_preparation(report, dataset, costs)
                verify_frozen_inputs(plan, args)
                report['experiment'] = {'frozen_plan_sha256': _hash(plan_path), 'cpu_threads': args.cpu_threads,
                                        'shared_cache_bytes': used, 'preparation_charged_to_each_model': True,
                                        'test_used_for_selection': False}
                path = output / f'{name}-{condition}.json'
                path.write_text(json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False) + '\n', encoding='utf-8')
                print('DONE', name, condition, json.dumps(report['metrics']), file=sys.stderr, flush=True)
    return plan


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--accept-research-license', action='store_true')
    for name in ('manifest', 'cmlr-dir', 'checkpoint', 'source-dir', 'output-dir'):
        p.add_argument('--' + name, required=True)
    p.add_argument('--adapter')
    p.add_argument('--selection', help='Frozen dev selection JSON; omission uses identity preprocessing')
    p.add_argument('--device', default='auto')
    p.add_argument('--ctc-weight', type=float, default=.1)
    p.add_argument('--reverse-weight', type=float, default=.3)
    p.add_argument('--reverse-scoring', choices=('sequential', 'batched'), default='sequential')
    p.add_argument('--reverse-validation', help='Real-checkpoint dev parity proof required for batched scoring')
    p.add_argument('--synthetic-stress', action='store_true')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--cpu-threads', type=int, default=4)
    p.add_argument('--cache-max-mib', type=int, default=256)
    args = p.parse_args(argv)
    if not args.accept_research_license:
        p.error('read the original model/data licenses and pass --accept-research-license for local research')
    if not 1 <= args.cpu_threads <= 32 or not 1 <= args.cache_max_mib <= 4096 or not 0 <= args.seed < 2 ** 32:
        p.error('invalid CPU thread count, image-cache budget or seed')
    if any(not math.isfinite(x) or not 0 <= x <= 1 for x in (args.ctc_weight, args.reverse_weight)):
        p.error('CTC/reverse weights must be finite within [0,1]')
    try:
        run(args)
    except (OSError, ValueError, KeyError) as exc:
        p.error(str(exc))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
