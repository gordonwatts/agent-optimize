"""Atomic accepted artifacts and best-effort persistent quarantine."""

import hashlib
from pathlib import Path
import re
import uuid

from .purity import validate_reusable_source
from .fast_adapter import adapter_source
from .store import json_bytes, read_json, session_lock, write_json


def contract_key(contract):
    return hashlib.sha256(json_bytes(contract)).hexdigest()[:24]


def manifest_seal(manifest):
    return hashlib.sha256(json_bytes({key: value for key, value in manifest.items() if key != "metadata_sha256"})).hexdigest()


def validate_manifest(manifest, artifact_id, *, allow_prospective=False):
    if not re.fullmatch(r"[0-9a-f]{32}", artifact_id) or manifest.get("artifact_id") != artifact_id:
        raise ValueError("artifact identity is invalid")
    states = {"accepted", "prospective"} if allow_prospective else {"accepted"}
    if manifest.get("state") not in states or manifest.get("metadata_sha256") != manifest_seal(manifest):
        raise ValueError("accepted artifact metadata is invalid or corrupt")
    for field in ("reference_fingerprint", "dependency_fingerprint", "environment", "contract", "correctness", "timing", "generator", "source_sha256"):
        if field not in manifest:
            raise ValueError(f"accepted artifact is missing {field}")
    if manifest["correctness"].get("status") != "passed" or not manifest["correctness"].get("case_count"):
        raise ValueError("artifact has no passing correctness evidence")
    if manifest["state"] == "accepted" and manifest["timing"].get("eligibility", {}).get("eligible") is not True:
        raise ValueError("artifact has no passing timing evidence")
    if manifest.get("dispatch_mode", "careful") not in ("fast", "careful"):
        raise ValueError("artifact dispatch mode is invalid")
    if manifest["timing"].get("dispatch_mode", manifest.get("dispatch_mode", "careful")) != manifest.get("dispatch_mode", "careful"):
        raise ValueError("artifact timing mode disagrees with its dispatch mode")


def install_artifact(root: Path, manifest, source, *, prospective=False):
    validate_reusable_source(source)
    root = root.resolve()
    artifact_id = manifest.setdefault("artifact_id", uuid.uuid4().hex)
    if not re.fullmatch(r"[0-9a-f]{32}", artifact_id) or not re.fullmatch(r"[0-9a-f]{24}", manifest["function_id"]):
        raise ValueError("artifact/function identity is invalid")
    manifest["schema_version"] = 1
    manifest["state"] = "prospective" if prospective else "accepted"
    manifest["source_sha256"] = hashlib.sha256(source.encode("utf-8")).hexdigest()
    adapter = adapter_source(manifest["contract"])
    manifest["adapter_sha256"] = hashlib.sha256(adapter.encode("utf-8")).hexdigest()
    evidence = manifest.pop("evaluation_evidence", {"schema_version": 1, "correctness": manifest["correctness"], "timing": manifest["timing"]})
    evidence_bytes = json_bytes(evidence)
    manifest["evidence_sha256"] = hashlib.sha256(evidence_bytes).hexdigest()
    manifest["metadata_sha256"] = manifest_seal(manifest)
    validate_manifest(manifest, artifact_id, allow_prospective=prospective)
    root.mkdir(parents=True, exist_ok=True)
    with session_lock(root):
        directory = root / "artifacts" / artifact_id
        pending = root / "artifacts" / f".pending-{artifact_id}"
        pending.mkdir(parents=True)
        (pending / "kernel.py").write_bytes(source.encode("utf-8"))
        (pending / "adapter.py").write_bytes(adapter.encode("utf-8"))
        (pending / "evidence.json").write_bytes(evidence_bytes)
        write_json(pending / "manifest.json", manifest)
        pending.replace(directory)
        index_path = root / "accepted" / manifest["function_id"] / "index.json"
        index_path.parent.mkdir(parents=True, exist_ok=True)
        if index_path.exists():
            index = read_json(index_path)
        else:
            index = {"schema_version": 1, "entries": {}}
        entry_key = contract_key(manifest["contract"]) + ":" + manifest.get("dispatch_mode", "careful")
        index["entries"][entry_key] = artifact_id
        write_json(index_path, index)
    return directory / "manifest.json"


def quarantine_path(root, artifact_id):
    return root / "quarantine" / f"{artifact_id}.json"


def persist_quarantine(root, artifact_id, phase, error):
    path = quarantine_path(root, artifact_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    write_json(path, {
        "schema_version": 1, "artifact_id": artifact_id, "phase": phase,
        "reason": f"{type(error).__name__}: artifact failed; reference Python retained",
    })
