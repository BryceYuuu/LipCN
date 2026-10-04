"""Benchmark provenance guards, using tiny local files and no model execution."""
import hashlib
import json
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace

import pytest

from research.evaluation import Dataset, Sample
from research.scripts import benchmark_chinese_models as benchmark


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")
    return hashlib.sha256(path.read_bytes()).hexdigest()


def choice(hashes):
    return {"schema_version": 1, "selection_partition": "dev", "test_used_for_selection": False,
            "selected_preprocessing": "identity", "input_sha256": hashes}


@pytest.mark.parametrize("hashes", [{}, [], {"dev.json": None}, {"dev.json": True},
                                   {"dev.json": ""}, {"dev.json": "a" * 63},
                                   {"dev.json": "a" * 65}, {"dev.json": "A" * 64},
                                   {"dev.json": "g" * 64}])
def test_selected_preset_rejects_missing_nonstring_or_malformed_report_hashes(tmp_path, hashes):
    path = tmp_path / "selection.json"
    write_json(path, choice(hashes))
    with pytest.raises(ValueError, match="hashes"):
        benchmark.selected_preset(path)


@pytest.mark.parametrize("change", [{"schema_version": 0}, {"schema_version": 2},
                                    {"schema_version": True}, {"schema_version": 1.0},
                                    {"selection_partition": "test"}, {"selection_partition": "train"},
                                    {"test_used_for_selection": True}, {"test_used_for_selection": 0},
                                    {"selected_preprocessing": "automatic"}])
def test_selected_preset_rejects_non_dev_or_unfrozen_choice(tmp_path, change):
    data = choice({"dev.json": "a" * 64})
    data.update(change)
    path = tmp_path / "selection.json"
    write_json(path, data)
    with pytest.raises(ValueError, match="dev-only"):
        benchmark.selected_preset(path)


@pytest.mark.parametrize("data", [None, [], "invalid", 1])
def test_selected_preset_requires_a_json_object(tmp_path, data):
    path = tmp_path / "selection.json"
    write_json(path, data)
    with pytest.raises(ValueError, match="dev-only"):
        benchmark.selected_preset(path)


@pytest.fixture
def dev_evidence(tmp_path):
    dataset = Dataset((Sample("test-a", "/local/test-a.mp4", "今天出门", "test-speaker", "test-session",
                              "verified label", True, "mouth_roi", True, sha256="a" * 64, split="test"),), split="test")
    report = {"samples": [{"sample_id": "dev-a", "split": "dev", "video_sha256": "b" * 64,
                            "speaker": "dev-speaker", "session": "dev-session",
                            "reference": "明天开会", "raw": "明天开会"}]}
    path = tmp_path / "dev-one.json"
    digest = write_json(path, report)
    selection_path = tmp_path / "selection.json"
    return dataset, report, path, digest, selection_path


@pytest.mark.parametrize("provenance_form", [False, True])
def test_selection_reports_binds_each_report_hash_in_both_supported_formats(dev_evidence, provenance_form):
    dataset, report, path, digest, selection_path = dev_evidence
    selection = ({"provenance": {"dev_report_sha256": {"one": digest}}} if provenance_form else
                 {"input_sha256": {"dev-one.json": digest}})
    assert benchmark.selection_reports(selection_path, selection, dataset) == {str(path.resolve()): digest}


@pytest.mark.parametrize("split", ["dev", "val", "validation"])
def test_development_split_aliases_are_accepted_and_report_is_frozen(dev_evidence, split):
    dataset, report, path, _, selection_path = dev_evidence
    report["samples"][0]["split"] = split
    digest = write_json(path, report)
    selection = {"input_sha256": {str(path): digest}}
    assert benchmark.selection_reports(selection_path, selection, dataset) == {str(path): digest}


@pytest.mark.parametrize("change", [{"split": "test"}, {"split": "train"}, {"split": None},
                                    {"sample_id": "test-a"}, {"video_sha256": "a" * 64}])
def test_selection_reports_rejects_test_partition_identity_or_source_video_hash(dev_evidence, change):
    dataset, report, path, _, selection_path = dev_evidence
    report["samples"][0].update(change)
    digest = write_json(path, report)
    with pytest.raises(ValueError, match="development-only and disjoint"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)


def test_selection_reports_rejects_changed_reference_even_when_ids_are_unchanged(dev_evidence):
    dataset, report, path, digest, selection_path = dev_evidence
    report["samples"][0]["reference"] = "今天开会"
    write_json(path, report)
    with pytest.raises(ValueError, match="changed after selection"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)


def test_selection_reports_rejects_empty_report_and_no_selection_is_explicit(dev_evidence):
    dataset, report, path, _, selection_path = dev_evidence
    digest = write_json(path, {"samples": []})
    with pytest.raises(ValueError, match="development-only"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)
    assert benchmark.selection_reports(None, None, dataset) == {}


@pytest.mark.parametrize("field", ["sample_id", "raw", "reference", "video_sha256"])
def test_development_report_requires_complete_prediction_identity(dev_evidence, field):
    dataset, report, path, _, selection_path = dev_evidence
    del report["samples"][0][field]
    digest = write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)


@pytest.mark.parametrize("change", [{"sample_id": ""}, {"sample_id": 1}, {"raw": None},
                                    {"reference": 1}, {"video_sha256": None},
                                    {"video_sha256": "b" * 63}, {"video_sha256": "B" * 64},
                                    {"video_sha256": "z" * 64}])
def test_development_report_rejects_malformed_identity_or_pixel_provenance(dev_evidence, change):
    dataset, report, path, _, selection_path = dev_evidence
    report["samples"][0].update(change)
    digest = write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)


def test_development_report_rejects_duplicate_rows(dev_evidence):
    dataset, report, path, _, selection_path = dev_evidence
    report["samples"].append(deepcopy(report["samples"][0]))
    digest = write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)


@pytest.mark.parametrize("report", [[], {"samples": {}}, {"samples": [None]},
                                   {"samples": [{"id": "dev-a", "split": "dev", "reference": "独立标注",
                                                 "sha256": "b" * 64, "video": "dev-a.mp4"}]}])
def test_development_evidence_must_be_a_prediction_report_not_a_manifest(dev_evidence, report):
    dataset, _, path, _, selection_path = dev_evidence
    digest = write_json(path, report)
    with pytest.raises(ValueError, match="development-only"):
        benchmark.selection_reports(selection_path, {"input_sha256": {str(path): digest}}, dataset)


@pytest.fixture
def frozen_inputs(tmp_path, monkeypatch):
    root = tmp_path / "checkout"
    cmlr = tmp_path / "cmlr"
    paths = {"manifest": tmp_path / "test.json", "checkpoint": tmp_path / "cnvsrc.pth",
             "selection": tmp_path / "selection.json", "adapter": tmp_path / "adapter.pth",
             "reverse_validation": tmp_path / "reverse-validation.json",
             "source": root / "research/example.py", "dev_report": tmp_path / "dev.json",
             "mean_face": root / "lipflow/mean_face.npy", "face_tracker": root / "models/face_landmarker.task"}
    cmlr_names = ("vsr/model.json", "vsr/model.pth", "lm/model.json", "lm/model.pth")
    paths.update({name: cmlr / name for name in cmlr_names})
    hashes = {}
    for key, path in paths.items():
        # These are file-integrity fixtures, not actual checkpoints or a dataset.
        hashes[key] = write_json(path, {"fixture": key, "reference": "原始独立标注"})
    args = SimpleNamespace(**{name: str(paths[name]) for name in
                              ("manifest", "checkpoint", "selection", "adapter", "reverse_validation")},
                           cmlr_dir=str(cmlr))
    plan = {"manifest_sha256": hashes["manifest"], "cnvsrc_checkpoint_sha256": hashes["checkpoint"],
            "selection_sha256": hashes["selection"], "adapter_sha256": hashes["adapter"],
            "reverse_validation_sha256": hashes["reverse_validation"],
            "source_sha256": {"research/example.py": hashes["source"]},
            "cmlr_file_sha256": {name: hashes[name] for name in cmlr_names},
            "development_report_sha256": {str(paths["dev_report"]): hashes["dev_report"]},
            "binary_asset_sha256": {str(paths[name]): hashes[name] for name in ("mean_face", "face_tracker")}}
    monkeypatch.setattr(benchmark, "REPOSITORY_ROOT", root)
    return paths, plan, args


def test_verify_frozen_inputs_accepts_exactly_bound_files(frozen_inputs):
    _, plan, args = frozen_inputs
    assert benchmark.verify_frozen_inputs(plan, args) is None


@pytest.mark.parametrize("changed", ["manifest", "checkpoint", "selection", "adapter", "reverse_validation",
                                     "source", "dev_report", "vsr/model.json", "vsr/model.pth",
                                     "lm/model.json", "lm/model.pth", "mean_face", "face_tracker"])
def test_verify_frozen_inputs_rejects_each_changed_dependency(frozen_inputs, changed):
    paths, plan, args = frozen_inputs
    data = json.loads(paths[changed].read_text(encoding="utf-8"))
    data["reference"] = "被改变的标注或文件内容"
    write_json(paths[changed], data)
    with pytest.raises(ValueError, match="frozen benchmark input changed"):
        benchmark.verify_frozen_inputs(plan, args)


def test_verify_frozen_inputs_rejects_deleted_bound_evidence(frozen_inputs):
    paths, plan, args = frozen_inputs
    paths["dev_report"].unlink()
    with pytest.raises(FileNotFoundError):
        benchmark.verify_frozen_inputs(plan, args)


def test_omitted_optional_models_and_selection_do_not_invent_hashes(frozen_inputs):
    _, plan, args = frozen_inputs
    args.selection = args.adapter = args.reverse_validation = None
    plan["selection_sha256"] = plan["adapter_sha256"] = plan["reverse_validation_sha256"] = None
    assert benchmark.verify_frozen_inputs(plan, args) is None


@pytest.fixture
def reverse_proof():
    source = {"research/reverse_batch.py": "c" * 64,
              "research/scripts/evaluate_cnvsrc.py": "d" * 64,
              "research/scripts/compare_chinese_decoders.py": "e" * 64}
    plan = {"cnvsrc_checkpoint_sha256": "a" * 64, "adapter_sha256": "b" * 64,
            "source_sha256": source,
            "arms": {"cnvsrc_candidate": {"reverse_tie_tolerance": 1e-3}}}
    args = SimpleNamespace(ctc_weight=.1, reverse_weight=.3)
    proof = {"schema_version": 1, "partition": "dev", "passed": True, "test_used": False,
             "samples": 2, "checkpoint_sha256": "a" * 64, "adapter_sha256": "b" * 64,
             "beam_size": 40, "ctc_weight": .1, "reverse_weight": .3, "nbest": 10,
             "tie_tolerance": 1e-3, "source_sha256": deepcopy(source),
             "rows": [{"sample_id": f"dev-{index}", "ranking_identical": True, "raw_identical": True,
                        "action_identical": True, "max_absolute_reverse_difference": 3.814697265625e-6}
                       for index in (1, 2)]}
    return proof, plan, args


def test_reverse_validation_accepts_complete_frozen_dev_parity(reverse_proof):
    proof, plan, args = reverse_proof
    assert benchmark.validate_reverse_validation(proof, plan, args) is None
    proof["adapter_sha256"] = plan["adapter_sha256"] = None
    assert benchmark.validate_reverse_validation(proof, plan, args) is None


@pytest.mark.parametrize("change", [{"schema_version": True}, {"schema_version": 1.0}, {"schema_version": 2},
                                    {"partition": "test"}, {"partition": "train"}, {"passed": False},
                                    {"passed": 1}, {"test_used": True}, {"test_used": 0},
                                    {"samples": True}, {"samples": 2.0}, {"samples": 0}, {"samples": 3},
                                    {"rows": []}, {"rows": None}, {"source_sha256": None},
                                    {"checkpoint_sha256": "f" * 64}, {"adapter_sha256": "f" * 64},
                                    {"ctc_weight": .3}, {"reverse_weight": .1}, {"beam_size": 4},
                                    {"nbest": 5}, {"tie_tolerance": .1}])
def test_reverse_validation_rejects_non_dev_partial_or_mismatched_proof(reverse_proof, change):
    proof, plan, args = reverse_proof
    proof.update(change)
    with pytest.raises(ValueError, match="batched validation"):
        benchmark.validate_reverse_validation(proof, plan, args)


@pytest.mark.parametrize("name", ["research/reverse_batch.py", "research/scripts/evaluate_cnvsrc.py",
                                 "research/scripts/compare_chinese_decoders.py"])
@pytest.mark.parametrize("changed", [False, True])
def test_reverse_validation_binds_batch_serial_and_integration_source(reverse_proof, name, changed):
    proof, plan, args = reverse_proof
    if changed:
        proof["source_sha256"][name] = "f" * 64
    else:
        del proof["source_sha256"][name]
    with pytest.raises(ValueError, match="frozen candidate"):
        benchmark.validate_reverse_validation(proof, plan, args)


@pytest.mark.parametrize("difference", [-1e-6, 1e-3, .5, float("nan"), float("inf"), True, "0", None])
def test_reverse_validation_rejects_invalid_or_unacceptably_large_score_difference(reverse_proof, difference):
    proof, plan, args = reverse_proof
    proof["rows"][0]["max_absolute_reverse_difference"] = difference
    with pytest.raises(ValueError, match="parity row"):
        benchmark.validate_reverse_validation(proof, plan, args)


@pytest.mark.parametrize("field", ["ranking_identical", "raw_identical", "action_identical"])
@pytest.mark.parametrize("value", [False, 1, None])
def test_reverse_validation_requires_explicit_true_for_every_parity_field(reverse_proof, field, value):
    proof, plan, args = reverse_proof
    proof["rows"][0][field] = value
    with pytest.raises(ValueError, match="parity row"):
        benchmark.validate_reverse_validation(proof, plan, args)


@pytest.mark.parametrize("row", [None, {}, {"sample_id": ""}])
def test_reverse_validation_requires_complete_rows(reverse_proof, row):
    proof, plan, args = reverse_proof
    proof["rows"][0] = row
    with pytest.raises(ValueError, match="parity row"):
        benchmark.validate_reverse_validation(proof, plan, args)


def test_reverse_validation_rejects_duplicate_sample_ids(reverse_proof):
    proof, plan, args = reverse_proof
    proof["rows"][1]["sample_id"] = proof["rows"][0]["sample_id"]
    with pytest.raises(ValueError, match="unique"):
        benchmark.validate_reverse_validation(proof, plan, args)
