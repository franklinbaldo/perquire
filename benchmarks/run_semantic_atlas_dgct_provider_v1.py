#!/usr/bin/env python3
"""Run the preregistered provider-backed Semantic Atlas DGCT v1.

Perquire supplies provider/caching/retry infrastructure. The gauge and SRF
implementation are imported from a separately checked-out, frozen `papers`
commit. The runner intentionally exposes no scientific defaults beyond the
versioned freeze file.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

from perquire.embeddings.openrouter_embeddings import OpenRouterEmbeddingProvider
from perquire.llm.openrouter_provider import OpenRouterProvider


ENVIRONMENT_PACKAGES = ("perquire", "litellm", "numpy", "openai", "httpx", "pydantic")


def sha256_bytes(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def environment_manifest() -> dict[str, Any]:
    packages: dict[str, str | None] = {}
    for package in ENVIRONMENT_PACKAGES:
        try:
            packages[package] = metadata.version(package)
        except metadata.PackageNotFoundError:
            packages[package] = None
    try:
        git_sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        git_sha = None
    return {
        "perquire_git_sha": git_sha,
        "python_version": platform.python_version(),
        "python_implementation": platform.python_implementation(),
        "platform": platform.platform(),
        "packages": packages,
    }


def git_paths(repo: Path, commit: str) -> list[str]:
    output = subprocess.check_output(
        ["git", "-C", str(repo), "ls-tree", "-r", "--name-only", commit], text=True
    )
    paths = [line.strip() for line in output.splitlines() if line.strip().endswith(".md")]
    return sorted(paths, key=lambda path: hashlib.sha256(path.encode()).hexdigest())


def git_text(repo: Path, commit: str, path: str, limit: int) -> str:
    raw = subprocess.check_output(["git", "-C", str(repo), "show", f"{commit}:{path}"])
    return raw.decode("utf-8", errors="replace").strip()[:limit]


def corpus_split(freeze: dict[str, Any], papers_repo: Path) -> tuple[list[str], list[str]]:
    cfg = freeze["corpus"]
    total = int(cfg["calibration_count"]) + int(cfg["heldout_count"]) + int(cfg["trajectory_count"])
    paths = git_paths(papers_repo, freeze["corpus_source_commit"])
    if len(paths) < total:
        raise RuntimeError(f"need {total} markdown paths, found {len(paths)}")
    chosen = paths[:total]
    cal_n = int(cfg["calibration_count"])
    held_n = int(cfg["heldout_count"])
    trajectory_start = cal_n + held_n
    return chosen[:cal_n], chosen[trajectory_start:total]


def normalize_rows(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, 1e-12)


def embeddings(provider: OpenRouterEmbeddingProvider, texts: list[str]) -> np.ndarray:
    rows = provider.embed_batch(texts)
    return normalize_rows(np.vstack([np.asarray(row.embedding, dtype=np.float64) for row in rows]))


def cumulative_states(prompt: str, completion: str, *, word_chunk: int, max_words: int) -> list[str]:
    words = completion.strip().split()
    if not words:
        raise RuntimeError("generator returned an empty completion")
    words = words[:max_words]
    states = [prompt]
    for end in range(word_chunk, len(words) + word_chunk, word_chunk):
        prefix = words[: min(end, len(words))]
        state = prompt + " " + " ".join(prefix)
        if state != states[-1]:
            states.append(state)
        if end >= len(words):
            break
    final = prompt + " " + " ".join(words)
    if final != states[-1]:
        states.append(final)
    if len(states) < 2:
        raise RuntimeError("trajectory produced fewer than two cumulative states")
    return states


def generate_trajectories(
    llm: OpenRouterProvider,
    prompts: list[str],
    cfg: dict[str, Any],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for prompt_index, source in enumerate(prompts):
        prompt = source + "\n\nContinue coherently:"
        response = llm.generate_response(
            prompt,
            cache_request_id=f"semantic-atlas-dgct-provider-v1:trajectory:{prompt_index}",
            temperature=float(cfg["temperature"]),
            max_tokens=int(cfg["max_tokens"]),
        )
        states = cumulative_states(
            prompt,
            response.content,
            word_chunk=int(cfg["word_chunk"]),
            max_words=int(cfg["max_words"]),
        )
        rows.append(
            {
                "prompt_index": prompt_index,
                "completion_sha256": hashlib.sha256(response.content.encode()).hexdigest(),
                "state_count": len(states),
                "texts": states,
            }
        )
    return rows


def flatten_trajectories(rows: list[dict[str, Any]]) -> tuple[list[str], list[tuple[int, int]]]:
    texts: list[str] = []
    spans: list[tuple[int, int]] = []
    for row in rows:
        start = len(texts)
        texts.extend(row["texts"])
        spans.append((start, len(texts)))
    return texts, spans


def split_trajectory_indices(count: int, *, seed: int, train_fraction: float) -> tuple[list[int], list[int]]:
    if count < 2:
        raise ValueError("need at least two trajectories")
    keys = [
        (hashlib.sha256(f"{seed}:{index}".encode()).hexdigest(), index)
        for index in range(count)
    ]
    ordered = [index for _, index in sorted(keys)]
    n_train = int(round(count * train_fraction))
    n_train = min(max(n_train, 1), count - 1)
    return sorted(ordered[:n_train]), sorted(ordered[n_train:])


def paths_from_states(states: np.ndarray, spans: list[tuple[int, int]]) -> list[np.ndarray]:
    return [states[start:end] for start, end in spans]


def samples(paths: list[np.ndarray], indices: list[int]) -> tuple[np.ndarray, np.ndarray]:
    state_rows: list[np.ndarray] = []
    delta_rows: list[np.ndarray] = []
    for index in indices:
        path = np.asarray(paths[index], dtype=np.float64)
        if len(path) < 2:
            raise RuntimeError("trajectory must contain at least two states")
        state_rows.append(path[:-1])
        delta_rows.append(np.diff(path, axis=0))
    return np.vstack(state_rows), np.vstack(delta_rows)


def public_report(report: dict[str, Any]) -> tuple[dict[str, Any], np.ndarray]:
    result = dict(report)
    result.pop("train_residuals", None)
    test_residual = np.asarray(result.pop("test_residuals"), dtype=np.float64)
    return result, test_residual


def concordance_delta(raw: dict[str, float], residual: dict[str, float]) -> dict[str, float]:
    return {
        "mean_row_cosine_gain": float(residual["mean_row_cosine"] - raw["mean_row_cosine"]),
        "normalized_rmse_ratio": float(
            residual["normalized_rmse"] / max(raw["normalized_rmse"], 1e-12)
        ),
        "pairwise_distance_correlation_gain": float(
            residual["pairwise_distance_correlation"] - raw["pairwise_distance_correlation"]
        ),
    }


def provider_from_cfg(cfg: dict[str, Any], *, cache_path: Path) -> OpenRouterEmbeddingProvider:
    return OpenRouterEmbeddingProvider(
        config={
            "model": cfg["model"],
            "requests_per_minute": int(cfg["requests_per_minute"]),
            "max_retries": int(cfg["max_retries"]),
            "cache_path": str(cache_path),
        }
    )


def run(freeze_path: Path, papers_repo: Path, output: Path) -> dict[str, Any]:
    freeze_raw = freeze_path.read_bytes()
    freeze = json.loads(freeze_raw)

    actual_harness = subprocess.check_output(
        ["git", "-C", str(papers_repo), "rev-parse", "HEAD"], text=True
    ).strip()
    if actual_harness != freeze["papers_harness_commit"]:
        raise RuntimeError(
            f"papers checkout {actual_harness} != frozen harness {freeze['papers_harness_commit']}"
        )

    atlas_src = papers_repo / "experiments" / "semantic_atlas" / "src"
    sys.path.insert(0, str(atlas_src))
    from semantic_atlas.dynamic_gauge import (  # type: ignore[import-not-found]
        FrozenFieldSpec,
        compression_report,
        paired_concordance,
    )
    from semantic_atlas.frame import QuasarFrame  # type: ignore[import-not-found]

    cal_paths, trajectory_paths = corpus_split(freeze, papers_repo)
    limit = int(freeze["corpus"]["excerpt_chars"])
    corpus_commit = freeze["corpus_source_commit"]
    cal_texts = [git_text(papers_repo, corpus_commit, path, limit) for path in cal_paths]
    prompt_texts = [git_text(papers_repo, corpus_commit, path, limit) for path in trajectory_paths]

    generator_cfg = freeze["generator"]
    llm = OpenRouterProvider(
        config={
            "model": generator_cfg["model"],
            "temperature": float(generator_cfg["temperature"]),
            "max_tokens": int(generator_cfg["max_tokens"]),
            "requests_per_minute": int(generator_cfg["requests_per_minute"]),
            "max_retries": int(generator_cfg["max_retries"]),
            "cache_mode": generator_cfg["cache_mode"],
        }
    )
    trajectories = generate_trajectories(llm, prompt_texts, generator_cfg)
    step_texts, spans = flatten_trajectories(trajectories)

    cache_root = Path(".cache/perquire/dgct-provider-v1")
    cache_root.mkdir(parents=True, exist_ok=True)
    reference_provider = provider_from_cfg(
        freeze["reference_observer"], cache_path=cache_root / "reference.sqlite"
    )
    transfer_provider = provider_from_cfg(
        freeze["transfer_observer"], cache_path=cache_root / "transfer.sqlite"
    )

    reference_cal = embeddings(reference_provider, cal_texts)
    transfer_cal = embeddings(transfer_provider, cal_texts)
    reference_step = embeddings(reference_provider, step_texts)
    transfer_step = embeddings(transfer_provider, step_texts)

    srf_dim = int(freeze["srf_dim"])
    reference_frame, canonical_targets = QuasarFrame.reference(reference_cal, dim=srf_dim)
    transfer_frame = QuasarFrame.fit(transfer_cal, canonical_targets)
    reference_states = reference_frame.canonical_vectors(reference_step)
    transfer_states = transfer_frame.canonical_vectors(transfer_step)
    reference_paths = paths_from_states(reference_states, spans)
    transfer_paths = paths_from_states(transfer_states, spans)

    split_cfg = freeze["trajectory_split"]
    train_indices, test_indices = split_trajectory_indices(
        len(trajectories),
        seed=int(split_cfg["seed"]),
        train_fraction=float(split_cfg["train_fraction"]),
    )
    ref_train_q, ref_train_f = samples(reference_paths, train_indices)
    ref_test_q, ref_test_f = samples(reference_paths, test_indices)
    tr_train_q, tr_train_f = samples(transfer_paths, train_indices)
    tr_test_q, tr_test_f = samples(transfer_paths, test_indices)
    if ref_test_f.shape != tr_test_f.shape:
        raise RuntimeError("paired observers yielded incompatible held-out dynamics")

    raw_cross = paired_concordance(ref_test_f, tr_test_f)
    amp = freeze["amplitude"]
    comp = freeze["complexity"]
    kwargs = {
        "amplitude_basis": amp["basis"],
        "amplitude_ridge": float(amp["ridge"]),
        "predictor_ridge": float(comp["predictor_ridge"]),
        "sample_fractions": tuple(float(x) for x in comp["sample_fractions"]),
        "target_relative_mse": float(comp["target_relative_mse"]),
    }

    field_results: dict[str, Any] = {}
    for raw_spec in freeze["fields"]:
        spec = FrozenFieldSpec(**raw_spec)
        ref_report, ref_residual = public_report(
            compression_report(
                ref_train_q,
                ref_train_f,
                ref_test_q,
                ref_test_f,
                reference_frame.quasars,
                spec,
                **kwargs,
            )
        )
        tr_report, tr_residual = public_report(
            compression_report(
                tr_train_q,
                tr_train_f,
                tr_test_q,
                tr_test_f,
                transfer_frame.quasars,
                spec,
                **kwargs,
            )
        )
        residual_cross = paired_concordance(ref_residual, tr_residual)
        key = f"{spec.family}:seed={spec.seed}:scale={spec.scale:g}"
        field_results[key] = {
            "field_spec": spec.to_dict(),
            "field_fingerprint": spec.fingerprint,
            "reference_observer": ref_report,
            "transfer_observer": tr_report,
            "cross_model": {
                "raw_test_dynamics": raw_cross,
                "test_residuals": residual_cross,
                "delta": concordance_delta(raw_cross, residual_cross),
            },
            "descriptive_checks": {
                "energy_lower_in_both_observers": bool(
                    ref_report["test_energy_ratio"] < 1.0 and tr_report["test_energy_ratio"] < 1.0
                ),
                "effective_rank_lower_in_both_observers": bool(
                    ref_report["test_effective_rank_ratio"] < 1.0
                    and tr_report["test_effective_rank_ratio"] < 1.0
                ),
                "cross_model_cosine_improves": bool(
                    residual_cross["mean_row_cosine"] > raw_cross["mean_row_cosine"]
                ),
                "cross_model_rmse_improves": bool(
                    residual_cross["normalized_rmse"] < raw_cross["normalized_rmse"]
                ),
            },
        }

    train_set = set(train_indices)
    payload = {
        "schema_version": 1,
        "experiment": freeze["experiment"],
        "freeze_sha256": sha256_bytes(freeze_raw),
        "environment": environment_manifest(),
        "papers_harness_commit": actual_harness,
        "corpus_source_commit": corpus_commit,
        "models": {
            "generator": llm.get_model_info(),
            "reference_observer": reference_provider.get_model_info(),
            "transfer_observer": transfer_provider.get_model_info(),
        },
        "corpus_paths": {
            "calibration": cal_paths,
            "trajectory": trajectory_paths,
        },
        "trajectory_split": {
            "train_indices": train_indices,
            "test_indices": test_indices,
            "train_step_count": len(ref_train_q),
            "test_step_count": len(ref_test_q),
        },
        "trajectories": [
            {
                "prompt_index": row["prompt_index"],
                "completion_sha256": row["completion_sha256"],
                "state_count": row["state_count"],
                "split": "train" if index in train_set else "test",
            }
            for index, row in enumerate(trajectories)
        ],
        "raw_cross_model_test_dynamics": raw_cross,
        "fields": field_results,
        "claim_boundary": freeze["claim_boundary"],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--freeze",
        type=Path,
        default=Path("docs/experiments/semantic_atlas_dgct_provider_v1.json"),
    )
    parser.add_argument("--papers-repo", type=Path, default=Path("papers-src"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("benchmark_results/semantic_atlas_dgct_provider_v1.json"),
    )
    args = parser.parse_args()
    result = run(args.freeze, args.papers_repo, args.output)
    summary = {
        key: {
            "reference_test_energy_ratio": value["reference_observer"]["test_energy_ratio"],
            "transfer_test_energy_ratio": value["transfer_observer"]["test_energy_ratio"],
            "reference_test_effective_rank_ratio": value["reference_observer"]["test_effective_rank_ratio"],
            "transfer_test_effective_rank_ratio": value["transfer_observer"]["test_effective_rank_ratio"],
            "cross_model_delta": value["cross_model"]["delta"],
        }
        for key, value in result["fields"].items()
    }
    print(json.dumps(summary, indent=2))
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
