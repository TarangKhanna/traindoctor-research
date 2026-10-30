"""The public packet must stand alone and reject coherently edited claims."""
import json
import shutil
from pathlib import Path

import pytest

from experiments.masked_mean_v1.public_packet import check, deidentify, digest

PACKET = Path(__file__).resolve().parents[1] / "evidence/masked_mean_v1"


@pytest.fixture
def public_copy(tmp_path, monkeypatch):
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    dst = tmp_path / "packet"
    shutil.copytree(PACKET, dst)
    return dst


def refresh_manifest(root, name):
    manifest = json.loads((root / "manifest.json").read_text())
    manifest[name] = digest((root / name).read_bytes())
    (root / "manifest.json").write_text(json.dumps(manifest))


def test_public_packet_recounts_without_private_receipts(monkeypatch):
    monkeypatch.setenv("TD_KEY", "state-recovery-demo-key")
    check(PACKET)


def test_changing_a_success_claim_fails_even_with_refreshed_file_hash(public_copy):
    p = public_copy / "packet.json"
    data = json.loads(p.read_text())
    data["controls"][0]["row"]["semantic_valid"] = False
    p.write_text(json.dumps(data))
    refresh_manifest(public_copy, p.name)
    with pytest.raises(ValueError, match="semantic success"):
        check(public_copy)


def test_model_metrics_are_recomputed_from_actions(public_copy):
    p = public_copy / "packet.json"
    data = json.loads(p.read_text())
    data["models"]["qwen"]["applied_edits"] = 1
    p.write_text(json.dumps(data))
    refresh_manifest(public_copy, p.name)
    with pytest.raises(ValueError, match="model action"):
        check(public_copy)


@pytest.mark.parametrize("remove", [("qwen",), ("gemma",), ("qwen", "gemma")])
def test_model_episodes_cannot_be_omitted_from_the_recount(public_copy, remove):
    p = public_copy / "packet.json"
    data = json.loads(p.read_text())
    for name in remove:
        del data["models"][name]
    p.write_text(json.dumps(data))
    refresh_manifest(public_copy, p.name)
    with pytest.raises(ValueError, match="model set"):
        check(public_copy)


def test_unmapped_host_path_is_rejected_even_with_refreshed_hash(public_copy):
    p = public_copy / "gemma_trajectory.json"
    data = json.loads(p.read_text())
    data["extra_path"] = "/Users/private-person/project"
    p.write_text(json.dumps(data))
    refresh_manifest(public_copy, p.name)
    with pytest.raises(ValueError, match="host path"):
        check(public_copy)


def test_packet_never_follows_a_symlink(public_copy, tmp_path):
    p = public_copy / "qwen_trajectory.json"
    outside = tmp_path / "outside.json"
    outside.write_bytes(p.read_bytes())
    p.unlink()
    p.symlink_to(outside)
    with pytest.raises(ValueError, match="symlink"):
        check(public_copy)


def test_diagnostic_cannot_be_semantically_corrected_after_the_model_run(public_copy):
    p = public_copy / "qwen_candidate.txt"
    p.write_text(p.read_text() + "\n# changed after generation\n")
    d = public_copy / "diagnostic.json"
    receipt = json.loads(d.read_text())
    receipt["candidate_sha256"] = digest(p.read_bytes())
    d.write_text(json.dumps(receipt))
    refresh_manifest(public_copy, p.name)
    refresh_manifest(public_copy, d.name)
    with pytest.raises(ValueError, match="verbatim model fragment"):
        check(public_copy)


def test_redaction_preserves_numbers_protocol_and_patch_text():
    original = {"path": "/Users/person/project", "reward": 0, "error": 9.44e-8,
                "raw": '<|tool_response>', "patch": 'loss = total / count.clamp_min(1)'}
    redacted = deidentify(original, [("/Users/person", "<HOME>")])
    assert redacted == {**original, "path": "<HOME>/project"}


def test_recount_never_falls_back_to_an_owner_key(monkeypatch):
    monkeypatch.delenv("TD_KEY", raising=False)
    with pytest.raises(ValueError, match="public demonstration key"):
        check(PACKET)
