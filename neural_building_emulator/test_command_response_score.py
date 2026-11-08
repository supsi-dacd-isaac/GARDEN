"""Command-response scoring must include successes as well as cooling failures."""

import jax
import jax.numpy as jnp
import numpy as np
import unittest

from neural_building_emulator.train import (
    estimate_event_response_energy_scales,
    event_response_energy_score,
)


def fixture():
    inputs = jnp.zeros((1, 24, 2)).at[:, 8:16, 1].set(1.0)
    truth = jnp.zeros((1, 24, 3)).at[:, 8:16, :].set(-1.0)
    return inputs, truth


def score(predictions, truth, inputs, *, commands=()):
    return event_response_energy_score(
        predictions, truth, inputs, jax.random.PRNGKey(13),
        horizon_steps=(2,), feature_scales=jnp.ones((1, 3)),
        events_per_profile=16, setpoint_delta_threshold=0.05,
        additional_event_inputs=commands,
    )


def test_force_off_loss_is_opt_in_and_detects_missed_drops():
    inputs, truth = fixture()
    predictions = jnp.zeros((1, 3, 24, 3))
    assert float(score(predictions, truth, inputs)) == 0.0
    assert float(score(predictions, truth, inputs, commands=((1, 0.5),))) > 0.5
    perfect = jnp.broadcast_to(truth[:, None], predictions.shape)
    assert float(score(perfect, truth, inputs, commands=((1, 0.5),))) < 0.01


def test_false_drops_are_penalized_even_when_observed_response_is_zero():
    inputs, dropped = fixture()
    truth = jnp.zeros_like(dropped)
    predictions = jnp.broadcast_to(dropped[:, None], (1, 3, 24, 3))
    assert float(score(predictions, truth, inputs, commands=((1, 0.5),))) > 0.5


def test_onset_and_release_are_both_events():
    inputs, _ = fixture()
    for change in (8, 16):
        truth = jnp.zeros((1, 24, 3)).at[:, change:, :].set(1.0)
        assert float(score(jnp.zeros((1, 3, 24, 3)), truth, inputs,
                           commands=((1, 0.5),))) > 0.1


def test_training_scales_match_event_definition_and_input_normalization():
    inputs, truth = fixture()
    kwargs = dict(horizon_steps=(2,), setpoint_delta_threshold=0.05)
    legacy = estimate_event_response_energy_scales(np.asarray(truth), np.asarray(inputs), **kwargs)
    physical = estimate_event_response_energy_scales(
        np.asarray(truth * 3), np.asarray(inputs), **kwargs, additional_event_inputs=((1, 0.5),))
    np.testing.assert_allclose(legacy, 1.0)
    np.testing.assert_allclose(physical, 3.0)
    predictions = jnp.zeros((1, 3, 24, 3))
    normalized = inputs.at[:, :, 1].set((inputs[:, :, 1] - 0.2) / 0.1)
    np.testing.assert_allclose(
        score(predictions, truth, inputs, commands=((1, 0.5),)),
        score(predictions, truth, normalized, commands=((1, 5.0),)),
    )


def test_constant_commands_have_zero_loss_and_finite_gradient():
    inputs, truth = fixture()
    predictions = jnp.zeros((1, 3, 24, 3))
    loss, gradient = jax.value_and_grad(
        lambda x: score(x, truth, jnp.zeros_like(inputs), commands=((1, 0.5),))
    )(predictions)
    assert float(loss) == 0.0
    assert np.isfinite(gradient).all()


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(
        unittest.FunctionTestCase(value)
        for name, value in sorted(globals().items()) if name.startswith("test_")
    )


if __name__ == "__main__":
    unittest.main()
