import pytest
import torch
import click

from architectures import MetadataLayout
from symqnet_cli import validate_checkpoint_metadata


def write_checkpoint(tmp_path, **overrides):
    layout = MetadataLayout.from_problem(n_qubits=10, M_evo=5)
    checkpoint = {
        "model_state_dict": {},
        "meta_dim": layout.meta_dim,
        "shots_encoding": None,
        "n_qubits": 10,
        "M_evo": 5,
        "rollout_steps": 10,
    }
    checkpoint.update(overrides)
    path = tmp_path / "checkpoint.pth"
    torch.save(checkpoint, path)
    return path


def test_metadata_layout_sizing():
    layout = MetadataLayout.from_problem(n_qubits=10, M_evo=5)
    assert layout.base == layout.n_qubits + 3 + layout.M_evo
    assert layout.theta_slot0 == layout.shots_slot + 1
    assert layout.meta_dim == layout.fisher_slot0 + layout.fisher_feat_dim


def test_validate_checkpoint_metadata_meta_dim_mismatch(tmp_path):
    path = write_checkpoint(tmp_path, meta_dim=999)
    with pytest.raises(click.ClickException, match="meta_dim mismatch"):
        validate_checkpoint_metadata(path)


def test_validate_checkpoint_metadata_version_pair(tmp_path):
    path = write_checkpoint(tmp_path, checkpoint_format="symqnet-ppo-v2")
    with pytest.raises(click.ClickException, match="checkpoint_format and checkpoint_version"):
        validate_checkpoint_metadata(path)


def test_validate_checkpoint_metadata_version_ok(tmp_path):
    path = write_checkpoint(
        tmp_path,
        checkpoint_format="symqnet-ppo-v2",
        checkpoint_version=1,
    )
    metadata = validate_checkpoint_metadata(path)
    assert metadata["checkpoint_format"] == "symqnet-ppo-v2"
    assert metadata["checkpoint_version"] == 1
