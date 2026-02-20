import pytest

from areal.api.cli_args import PPOActorConfig


class TestPPOActorConfigEngineIS:
    """Test PPOActorConfig validation for enable_MIS_TIS_correction."""

    def test_engine_is_requires_decoupled_or_recompute(self):
        """Test that enable_MIS_TIS_correction=True requires decoupled or recompute."""
        # Should raise when enable_MIS_TIS_correction=True without decoupled or recompute
        with pytest.raises(ValueError, match="enable_MIS_TIS_correction=True requires"):
            PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                enable_MIS_TIS_correction=True,
                use_decoupled_loss=False,
                prox_logp_method="loglinear",  # not recompute
            )

    def test_engine_is_works_with_decoupled(self):
        """Test that enable_MIS_TIS_correction works with decoupled loss."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
            enable_MIS_TIS_correction=True,
            use_decoupled_loss=True,
        )
        assert config.enable_MIS_TIS_correction is True

    def test_engine_is_works_with_recompute(self):
        """Test that enable_MIS_TIS_correction works with recompute."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
            enable_MIS_TIS_correction=True,
            use_decoupled_loss=False,
            prox_logp_method="recompute",
        )
        assert config.enable_MIS_TIS_correction is True

    def test_engine_is_defaults(self):
        """Test default values for engine_is parameters."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
        )
        assert config.enable_MIS_TIS_correction is False
        assert config.engine_mismatch_IS_mode == "sequence_mask"
        assert config.engine_mismatch_IS_cap == 3.0
