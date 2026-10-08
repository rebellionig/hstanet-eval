import json
from pathlib import Path

import pytest

from hstanet_eval.cli import main

EXAMPLES = Path(__file__).resolve().parent.parent / "examples"


def test_metrics_command_on_example_data(capsys):
    code = main(
        [
            "metrics",
            "--pred",
            str(EXAMPLES / "predictions.json"),
            "--gt",
            str(EXAMPLES / "ground_truth.json"),
            "--iou",
            "0.5",
            "0.75",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["iou_thresholds"] == [0.5, 0.75]
    assert 0.0 <= out["mAP"] <= 1.0
    assert out["recall"] == pytest.approx(1.0)


def test_latency_command_with_demo_target(capsys):
    code = main(
        [
            "latency",
            "--target",
            "hstanet_eval.demo:dummy_model",
            "--shape",
            "4,3,32,32",
            "--warmup",
            "2",
            "--runs",
            "5",
        ]
    )
    out = json.loads(capsys.readouterr().out)
    assert code == 0
    assert out["n"] == 5
    assert out["fps"] > 0


def test_latency_target_must_have_attribute():
    with pytest.raises(SystemExit):
        main(["latency", "--target", "hstanet_eval.demo"])


def test_version_flag(capsys):
    with pytest.raises(SystemExit) as exc:
        main(["--version"])
    assert exc.value.code == 0
    assert "0.1.0" in capsys.readouterr().out
