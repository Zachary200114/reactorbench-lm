from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch

from reactorbench.dataset import VerifiedDevelopmentCandidateArtifact
from reactorbench.evaluation.config import Phase6TestFreezeConfig, load_phase5_config
from reactorbench.evaluation.data import ExperimentExample, materialize_phase6_data
from reactorbench.evaluation.decoding import (
    DecodedPrediction,
    greedy_decode_predictions,
    parse_structured_prediction,
)
from reactorbench.evaluation.metrics import (
    bootstrap_mean_interval,
    calibration_metrics,
    set_f1_metrics,
)
from reactorbench.model import TransformerConfig, initialized_model
from reactorbench.schemas.base import canonical_sha256
from reactorbench.schemas.enums import ActionLabel, SplitName, TaskName
from reactorbench.schemas.target import NextActionTarget
from reactorbench.tokenizer import ProjectTokenizer


def _example() -> ExperimentExample:
    return ExperimentExample(
        example_id="example:phase6",
        split_name=SplitName.IID_TEST,
        task_name=TaskName.NEXT_ACTION,
        prompt_text="fictional bounded evidence",
        target_text=('{"immediate_action":"CONTINUE_MONITORING","task_name":"next_action"}'),
        classification_label="CONTINUE_MONITORING",
        source_checksum_sha256="a" * 64,
    )


def _tokenizer(monkeypatch: pytest.MonkeyPatch) -> ProjectTokenizer:
    tokenizer = object.__new__(ProjectTokenizer)
    monkeypatch.setattr(
        ProjectTokenizer,
        "encode",
        lambda _self, _text, *, add_bos=True, add_eos=True: (
            *((1,) if add_bos else ()),
            4,
            5,
            *((2,) if add_eos else ()),
        ),
    )
    monkeypatch.setattr(ProjectTokenizer, "decode", lambda _self, _ids: "not-json")
    return tokenizer


def test_strict_prediction_parser_rejects_wrong_tasks_and_duplicate_keys() -> None:
    valid = '{"immediate_action":"CONTINUE_MONITORING","task_name":"next_action"}'
    parsed, schema_valid, target, canonical = parse_structured_prediction(
        valid, task_name=TaskName.NEXT_ACTION
    )
    assert parsed
    assert schema_valid
    assert target is not None
    assert canonical == valid

    assert parse_structured_prediction(valid, task_name=TaskName.FAULT_FAMILY)[:2] == (
        True,
        False,
    )
    duplicate = (
        '{"task_name":"next_action","task_name":"next_action",'
        '"immediate_action":"CONTINUE_MONITORING"}'
    )
    assert parse_structured_prediction(duplicate, task_name=TaskName.NEXT_ACTION)[:2] == (
        False,
        False,
    )
    assert parse_structured_prediction("[]", task_name=TaskName.NEXT_ACTION)[:2] == (
        True,
        False,
    )
    assert parse_structured_prediction(
        '{"task_name":"next_action","immediate_action":"NOT_A_LABEL"}',
        task_name=TaskName.NEXT_ACTION,
    )[:2] == (True, False)
    assert parse_structured_prediction("NaN", task_name=TaskName.NEXT_ACTION)[:2] == (
        False,
        False,
    )
    with pytest.raises(TypeError, match="exact text"):
        parse_structured_prediction(cast(Any, b"{}"), task_name=TaskName.NEXT_ACTION)


def test_cached_greedy_decoder_is_bounded_and_deterministic(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = initialized_model(
        TransformerConfig(
            model_version="0.1.0",
            layers=1,
            width=32,
            heads=4,
            context_length=16,
            feed_forward_multiplier=2,
            dropout=0.0,
            tie_embeddings=True,
            bias=True,
        ),
        vocab_size=64,
        seed=77,
    )
    tokenizer = _tokenizer(monkeypatch)
    serialization = load_phase5_config(
        Path("configs/experiments/phase5-pilot-v0.1.0.toml")
    ).serialization
    first = greedy_decode_predictions(
        model,
        tokenizer,
        (_example(),),
        serialization,
        maximum_generated_tokens=3,
        batch_size=1,
        device=torch.device("cpu"),
    )
    second = greedy_decode_predictions(
        model,
        tokenizer,
        (_example(),),
        serialization,
        maximum_generated_tokens=3,
        batch_size=1,
        device=torch.device("cpu"),
    )
    assert first == second
    assert len(first) == 1
    assert first[0].generated_token_count <= 3
    assert not first[0].schema_valid
    assert first[0].confidence == 0.0
    with pytest.raises(ValueError, match="decoder token"):
        greedy_decode_predictions(
            model,
            tokenizer,
            (_example(),),
            serialization,
            maximum_generated_tokens=16,
            batch_size=1,
            device=torch.device("cpu"),
        )


def test_bootstrap_calibration_and_set_metrics_are_deterministic() -> None:
    first = bootstrap_mean_interval(
        (1.0, 0.0, 1.0, 1.0),
        resamples=2000,
        seed=6602,
        confidence_level=0.95,
    )
    second = bootstrap_mean_interval(
        (1.0, 0.0, 1.0, 1.0),
        resamples=2000,
        seed=6602,
        confidence_level=0.95,
    )
    assert first == second
    assert first.estimate == 0.75
    calibration = calibration_metrics(
        (True, False, True, True),
        (0.9, 0.8, 0.7, 0.6),
        bin_count=10,
        selective_coverage=0.5,
    )
    assert calibration.selective_coverage == 0.5
    assert calibration.selective_risk == 0.5
    sets = set_f1_metrics((("A", "B"), ("C",)), (("A",), ("C", "D")))
    assert sets.true_positive == 2
    assert sets.false_positive == 1
    assert sets.false_negative == 1
    assert sets.f1 == pytest.approx(2 / 3)
    empty_sets = set_f1_metrics(((),), ((),))
    assert empty_sets.precision == 1.0
    assert empty_sets.recall == 1.0
    with pytest.raises(TypeError, match="non-empty exact float"):
        bootstrap_mean_interval(
            cast(Any, [1.0]),
            resamples=2000,
            seed=6602,
            confidence_level=0.95,
        )
    with pytest.raises(ValueError, match="configuration"):
        bootstrap_mean_interval(
            (1.0,),
            resamples=99,
            seed=6602,
            confidence_level=0.95,
        )
    with pytest.raises(TypeError, match="aligned exact tuples"):
        calibration_metrics(
            (True,),
            (1.5,),
            bin_count=10,
            selective_coverage=0.8,
        )
    with pytest.raises(ValueError, match="bin_count"):
        calibration_metrics(
            (True,),
            (0.5,),
            bin_count=1,
            selective_coverage=0.8,
        )
    with pytest.raises(ValueError, match="set metrics"):
        set_f1_metrics((), ())


def test_prediction_contract_rejects_inconsistent_validity_and_confidence() -> None:
    payload = {
        "example_id": "example:test",
        "task_name": TaskName.NEXT_ACTION,
        "generated_text": "{}",
        "generated_token_count": 1,
        "prompt_truncated": False,
        "generation_truncated": False,
        "json_parse_success": False,
        "schema_valid": True,
        "predicted_target_json": "{}",
        "classification_label": None,
        "confidence": 0.5,
        "checksum_sha256": "0" * 64,
    }
    with pytest.raises(ValueError, match="must first parse"):
        DecodedPrediction.model_validate(payload)
    payload["json_parse_success"] = True
    payload["schema_valid"] = False
    with pytest.raises(ValueError, match="must exist exactly"):
        DecodedPrediction.model_validate(payload)
    payload["predicted_target_json"] = None
    payload["classification_label"] = "CONTINUE_MONITORING"
    with pytest.raises(ValueError, match="cannot carry"):
        DecodedPrediction.model_validate(payload)


def test_phase6_materializer_reconstructs_and_freezes_every_split() -> None:
    counts = {
        SplitName.IID_TRAIN: 630,
        SplitName.IID_VALIDATION: 252,
        SplitName.IID_TEST: 252,
        SplitName.TEMPLATE_TEST: 252,
        SplitName.COMPONENT_TEST: 72,
        SplitName.SEVERITY_TEST: 52,
        SplitName.COMPOSITION_TEST: 116,
        SplitName.COUNTERFACTUAL_TEST: 102,
        SplitName.NOISE_TEST: 48,
    }
    renders: list[SimpleNamespace] = []
    examples: list[SimpleNamespace] = []
    inventory: dict[SplitName, list[tuple[str, str, str]]] = {split: [] for split in SplitName}
    index = 0
    target = NextActionTarget(immediate_action=ActionLabel.CONTINUE_MONITORING)
    for split in SplitName:
        for _ in range(counts[split]):
            example_id = f"example-{index:04d}"
            render_id = f"render-{index:04d}"
            checksum = f"{index + 1:064x}"
            renders.append(
                SimpleNamespace(
                    render_id=render_id,
                    split_name=split,
                    text=f"fictional prompt {index}",
                    model_input_sha256="a" * 64,
                    text_sha256="b" * 64,
                )
            )
            examples.append(
                SimpleNamespace(
                    example_id=example_id,
                    split_name=split,
                    task_name=TaskName.NEXT_ACTION,
                    prompt_render_ids=(render_id,),
                    corruption_ids=(),
                    task_target=SimpleNamespace(target=target),
                    checksum_sha256=checksum,
                )
            )
            inventory[split].append((example_id, TaskName.NEXT_ACTION.value, checksum))
            index += 1
    hashes = {split: canonical_sha256(tuple(rows)) for split, rows in inventory.items()}
    freeze = Phase6TestFreezeConfig(
        split_manifest_raw_sha256="1" * 64,
        task_examples_raw_sha256="2" * 64,
        train_example_count=630,
        validation_example_count=252,
        test_example_count=894,
        component_test_inventory_sha256=hashes[SplitName.COMPONENT_TEST],
        composition_test_inventory_sha256=hashes[SplitName.COMPOSITION_TEST],
        counterfactual_test_inventory_sha256=hashes[SplitName.COUNTERFACTUAL_TEST],
        iid_test_inventory_sha256=hashes[SplitName.IID_TEST],
        noise_test_inventory_sha256=hashes[SplitName.NOISE_TEST],
        severity_test_inventory_sha256=hashes[SplitName.SEVERITY_TEST],
        template_test_inventory_sha256=hashes[SplitName.TEMPLATE_TEST],
        access_policy="after_owner_golden_approval_and_implementation_commit",
    )
    verified = VerifiedDevelopmentCandidateArtifact(
        manifest=cast(Any, None),
        metadata=cast(Any, None),
        candidate=cast(
            Any,
            SimpleNamespace(
                rendered_candidates=tuple(renders),
                corruption_records=(),
                task_examples=tuple(examples),
            ),
        ),
    )
    data = materialize_phase6_data(
        verified,
        freeze=freeze,
        maximum_prompt_utf8_bytes=1024,
    )
    assert len(data.all_records) == 1776
    assert len(data.by_split[SplitName.IID_TEST]) == 252
    assert data.inventory_sha256_by_split[SplitName.NOISE_TEST] == hashes[SplitName.NOISE_TEST]

    changed = freeze.model_copy(update={"iid_test_inventory_sha256": "0" * 64})
    with pytest.raises(ValueError, match="iid_test checksum"):
        materialize_phase6_data(
            verified,
            freeze=changed,
            maximum_prompt_utf8_bytes=1024,
        )
