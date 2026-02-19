import pytest

from areal.api.cli_args import PPOActorConfig


class TestPPOActorConfigEngineIS:
    """Test PPOActorConfig validation for engine_is_correction."""

    def test_engine_is_requires_decoupled_or_recompute(self):
        """Test that engine_is_correction=True requires decoupled or recompute."""
        # Should raise when engine_is_correction=True without decoupled or recompute
        with pytest.raises(ValueError, match="engine_is_correction=True requires"):
            PPOActorConfig(
                experiment_name="test",
                trial_name="test",
                path="/test/path",
                engine_is_correction=True,
                use_decoupled_loss=False,
                prox_logp_method="loglinear",  # not recompute
            )

    def test_engine_is_works_with_decoupled(self):
        """Test that engine_is_correction works with decoupled loss."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
            engine_is_correction=True,
            use_decoupled_loss=True,
        )
        assert config.engine_is_correction is True

    def test_engine_is_works_with_recompute(self):
        """Test that engine_is_correction works with recompute."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
            engine_is_correction=True,
            use_decoupled_loss=False,
            prox_logp_method="recompute",
        )
        assert config.engine_is_correction is True

    def test_engine_is_defaults(self):
        """Test default values for engine_is parameters."""
        config = PPOActorConfig(
            experiment_name="test",
            trial_name="test",
            path="/test/path",
        )
        assert config.engine_is_correction is False
        assert config.engine_is_mode == "sequence_mask"
        assert config.engine_is_cap == 3.0
