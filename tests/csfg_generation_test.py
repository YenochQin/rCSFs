"""Direct generator API/CLI contract; fixtures require no GRASPG executable."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from rcsfs import CsfgGenerationStats, generate_csfgs
from rcsfs._publication import PartialPublicationError
from rcsfs.cli import main

FIXTURES = Path(__file__).parent / "fixtures/csfg"
INPUT = """[csfggenerate]
inactive_core = 0
reference_configuration = ["1s(2,*)"]
labeling_space = "1s"
active_space = "3s,3p"
j_min = 0
j_max = 4
excitations = 2
"""


def generate(tmp_path: Path, **kwargs: object) -> CsfgGenerationStats:
    options = {
        "inactive_core": 0,
        "reference_configuration": ["1s(2,*)"],
        "labeling_space": "1s",
        "active_space": "3s,3p",
        "j_min": 0,
        "j_max": 4,
        "excitations": 2,
        "rcsfg_out": tmp_path / "rcsfg.out",
        "rlabel_out": tmp_path / "rlabel.out",
    }
    options.update(kwargs)
    return generate_csfgs(**options)


def test_csfg_public_api_golden_and_replacement(tmp_path: Path) -> None:
    stats = generate(tmp_path, threads=2)
    assert stats["record_count"] == 16
    assert stats["expanded_csf_count"] == 28
    assert stats["label_csf_count"] == 1
    assert stats["generator_count"] == 15
    assert all(stats["type_counts"])
    for name in ["rcsfg", "rlabel"]:
        assert (tmp_path / f"{name}.out").read_bytes() == (
            FIXTURES / f"{name}.txt"
        ).read_bytes()
    with pytest.raises(FileExistsError):
        generate(tmp_path)
    assert generate(tmp_path, overwrite=True)["record_count"] == 16
    assert not list(tmp_path.glob("rcsfs-csfg-*"))


@pytest.mark.parametrize(
    "options",
    [
        {"threads": True},
        {"threads": 0},
        {"excitations": 3},
        {"reference_configuration": "1s(2,*)"},
        {"reference_configuration": ["1λ(2,*)"]},
        {"inactive_core": False},
        {"labeling_space": "1s,2s"},
        {"j_min": 1},
        {"overwrite": 1},
        {"reference_configuration": ["2s(2,*)"]},
    ],
)
def test_invalid_models_leave_no_outputs(
    tmp_path: Path, options: dict[str, object]
) -> None:
    with pytest.raises((ValueError, TypeError)):
        generate(tmp_path, **options)
    assert not list(tmp_path.iterdir())


def test_aliases_and_existing_label_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="distinct"):
        generate(tmp_path, rlabel_out=tmp_path / "rcsfg.out")
    (tmp_path / "rlabel.out").write_text("preserve")
    with pytest.raises(FileExistsError):
        generate(tmp_path)
    assert not (tmp_path / "rcsfg.out").exists()
    assert (tmp_path / "rlabel.out").read_text() == "preserve"


def test_partial_publication_is_not_reported_success(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import rcsfs._publication as publication

    original = publication._publish_one

    def fail_second(source: Path, target: Path) -> None:
        if target.name == "rlabel.out":
            raise OSError("simulated label publication failure")
        original(source, target)

    monkeypatch.setattr(publication, "_publish_one", fail_second)
    with pytest.raises(PartialPublicationError) as caught:
        generate(tmp_path)
    assert caught.value.published == [tmp_path / "rcsfg.out"]
    assert (tmp_path / "rcsfg.out").exists()
    assert not list(tmp_path.glob("rcsfs-csfg-*"))


def test_cli_config_and_json(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(tmp_path)
    Path("rcsfs.toml").write_text(INPUT)
    assert main(["csfggenerate", "-c", "--threads", "2", "--json"]) == 0
    stats = json.loads(capsys.readouterr().out)
    assert stats["expanded_csf_count"] == 28
    assert Path("rlabel.out").read_bytes() == (FIXTURES / "rlabel.txt").read_bytes()


@pytest.mark.parametrize(
    "suffix",
    [
        "generate_descriptors = true\n",
        "threads = true\n",
        'rcsfg_out = "rcsfs.toml"\n',
        'rlabel_out = "rcsfg.out"\n',
        'overwrite = "yes"\n',
    ],
)
def test_cli_rejects_unsupported_or_unsafe_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, suffix: str
) -> None:
    monkeypatch.chdir(tmp_path)
    config = Path("rcsfs.toml")
    config.write_text(INPUT + suffix)
    before = config.read_bytes()
    assert main(["csfggenerate", "--json"]) == 1
    assert config.read_bytes() == before
    assert not Path("rcsfg.out").exists()
