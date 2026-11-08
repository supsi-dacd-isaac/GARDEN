"""Scientific invariants of the matched thermal noise ablation."""
import tempfile
from pathlib import Path
import unittest

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import optax

from neural_building_emulator_refactor.models.thermal_noise import ThermalNoiseModel, distribution_loss
from neural_building_emulator_refactor.thermal_noise_experiment import (
    aggregate_trace_scores, crps_array, save_eqx, score_samples, should_stop,
)


class ThermalNoiseTests(unittest.TestCase):
    def model(self, noise="exogenous", architecture="diagonal", truncate=4,
              constraint="contractive", matched=False):
        return ThermalNoiseModel(3,4,architecture=architecture,noise_conditioning=noise,
            state_dim=3,encoded_dim=2,latent_dim=2,width=8,truncate_steps=truncate,
            transition_constraint=constraint,matched_pole_initialization=matched,
            key=jax.random.PRNGKey(13))

    def test_contraction_ablation_matches_initial_values_and_sensitivities(self):
        a=self.model(matched=True)
        b=self.model(constraint="none",matched=True)
        for aa,bb in zip(jax.tree.leaves(eqx.filter(a,eqx.is_array)),jax.tree.leaves(eqx.filter(b,eqx.is_array))):
            np.testing.assert_array_equal(aa,bb)
        for i in range(4):
            m=jax.random.normal(jax.random.PRNGKey(i),(3,))
            z=jax.random.normal(jax.random.PRNGKey(i+10),(2,))
            for aa,bb in zip(a.coefficients(m,z),b.coefficients(m,z)):
                np.testing.assert_array_equal(aa,bb)
        raw=a.base_parameters[:a.state_dim]
        np.testing.assert_allclose(jax.jacfwd(a.pole_values)(raw),jax.jacfwd(b.pole_values)(raw),rtol=2e-6)
        # Both sides of the join converge to the same derivative.
        np.testing.assert_allclose(jax.jacfwd(a.pole_values)(raw),jax.jacfwd(b.pole_values)(raw+1e-5),rtol=2e-5)
        x=jax.random.normal(jax.random.PRNGKey(20),(40,4))
        for model in (a,b):
            self.assertTrue(np.isfinite(model.sample(m,x,jnp.asarray(.1),jax.random.PRNGKey(21),4)).all())
        np.testing.assert_array_equal(a.sample(m,x,jnp.asarray(.1),jax.random.PRNGKey(21),4),
                                      b.sample(m,x,jnp.asarray(.1),jax.random.PRNGKey(21),4))

    def test_unconstrained_poles_can_exceed_one_and_grow_common_noise_difference(self):
        bounded=self.model(matched=True,truncate=0)
        relaxed=self.model(constraint="none",matched=True,truncate=0)
        for offset in (-100.,100.):
            raw=bounded.base_parameters[:bounded.state_dim]+offset
            self.assertTrue(np.all(np.asarray(bounded.pole_values(raw))<1))
            self.assertTrue(np.all(np.asarray(relaxed.pole_values(raw))>0))
        self.assertTrue(np.all(np.asarray(relaxed.pole_values(raw))>1))
        relaxed=eqx.tree_at(lambda n:n.base_parameters,relaxed,
                            relaxed.base_parameters.at[:relaxed.state_dim].add(100.))
        m,z,x,eps=jnp.zeros(3),jnp.zeros(2),jnp.zeros((4,4)),jnp.zeros((4,3))
        delta=jnp.ones(3)*.01
        h1,_=relaxed.rollout(m,x,jnp.asarray(0.),z,eps,carry=jnp.zeros(3))
        h2,_=relaxed.rollout(m,x,jnp.asarray(0.),z,eps,carry=delta)
        rho,*_=relaxed.coefficients(m,z)
        np.testing.assert_allclose(h2-h1,rho**len(x)*delta,rtol=2e-5,atol=1e-5)
        self.assertTrue(np.all(np.asarray(h2-h1)>np.asarray(delta)))

    def test_ablation_changes_only_state_mask(self):
        a,b=self.model(),self.model("state")
        for aa,bb in zip(jax.tree.leaves(eqx.filter(a,eqx.is_array)),jax.tree.leaves(eqx.filter(b,eqx.is_array))):
            np.testing.assert_array_equal(aa,bb)
        m,z,u,h=jnp.zeros(3),jnp.zeros(2),jnp.zeros(4),jnp.ones(3)
        np.testing.assert_array_equal(a.noise_scale(m,z,u,h),a.noise_scale(m,z,u,h*10))
        jac=jax.jacfwd(lambda hh:b.noise_scale(m,z,u,hh))(h)
        self.assertGreater(float(jnp.linalg.norm(jac)),1e-7)
        # The scale regularisation issue in the older trainer cannot recur:
        # its trainable base scale participates in both active noise formulas.
        shifted=eqx.tree_at(lambda n:n.raw_process_scale,a,a.raw_process_scale+1)
        self.assertGreater(float(jnp.linalg.norm(shifted.noise_scale(m,z,u,h)-a.noise_scale(m,z,u,h))),1e-4)

    def test_exogenous_common_noise_difference_is_exact_diagonal_power(self):
        model=self.model(truncate=0)
        m,z=jnp.zeros(3),jnp.array([.2,-.4])
        x=jax.random.normal(jax.random.PRNGKey(1),(37,4))
        eps=jax.random.normal(jax.random.PRNGKey(2),(37,3))
        h=jnp.array([.2,.3,-.1]);delta=jnp.array([.1,-.1,.05])
        a,_=model.rollout(m,x,jnp.asarray(0.),z,eps,carry=h)
        b,_=model.rollout(m,x,jnp.asarray(0.),z,eps,carry=h+delta)
        rho,*_=model.coefficients(m,z)
        np.testing.assert_allclose(b-a,rho**len(x)*delta,atol=2e-6,rtol=2e-4)
        self.assertTrue(np.all(np.asarray(rho)<1))

    def test_tbptt_and_chunking_preserve_forward_rollout(self):
        for architecture in ("diagonal","lstm"):
            a=self.model("state",architecture,truncate=4)
            b=self.model("state",architecture,truncate=0)
            m,z=jnp.ones(3)*.1,jnp.ones(2)*.2
            x=jax.random.normal(jax.random.PRNGKey(3),(17,4))
            eps=jax.random.normal(jax.random.PRNGKey(4),(17,3))
            _,ya=a.rollout(m,x,jnp.asarray(.3),z,eps)
            _,yb=b.rollout(m,x,jnp.asarray(.3),z,eps)
            carry,y1=a.rollout(m,x[:8],jnp.asarray(.3),z,eps[:8])
            _,y2=a.rollout(m,x[8:],jnp.asarray(0.),z,eps[8:],carry=carry)
            np.testing.assert_allclose(ya,yb,atol=1e-6)
            np.testing.assert_allclose(ya,jnp.concatenate((y1,y2)),atol=1e-6)

    def test_all_variants_have_finite_trainable_stochastic_objective(self):
        for noise,architecture,constraint in (("exogenous","diagonal","contractive"),
                ("exogenous","diagonal","none"),("state","diagonal","contractive"),("state","lstm","none")):
            model=self.model(noise,architecture,constraint=constraint,matched=True)
            def loss(n):
                prediction=n.sample(jnp.zeros(3),jnp.ones((16,4))*.2,jnp.asarray(.1),jax.random.PRNGKey(5),4)
                return distribution_loss(prediction[None],jnp.zeros((1,16)))[0]
            value,grad=eqx.filter_value_and_grad(loss)(model)
            self.assertTrue(np.isfinite(value))
            self.assertTrue(all(np.isfinite(x).all() for x in jax.tree.leaves(eqx.filter(grad,eqx.is_array))))
            self.assertGreater(float(jnp.linalg.norm(grad.raw_process_scale)),0)

    def test_optimizer_resume_preserves_next_update(self):
        model=self.model();optim=optax.adam(.001)
        state=optim.init(eqx.filter(model,eqx.is_inexact_array))
        def update(m,s):
            grad=jax.tree.map(jnp.ones_like,eqx.filter(m,eqx.is_inexact_array))
            changes,s=optim.update(grad,s,eqx.filter(m,eqx.is_inexact_array))
            return eqx.apply_updates(m,changes),s
        model,state=update(model,state)
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/"resume.eqx"
            save_eqx(p,(model,state))
            loaded,loaded_state=eqx.tree_deserialise_leaves(p,(self.model(),optim.init(eqx.filter(self.model(),eqx.is_inexact_array))))
            a,sa=update(model,state);b,sb=update(loaded,loaded_state)
            for aa,bb in zip(jax.tree.leaves(eqx.filter((a,sa),eqx.is_array)),jax.tree.leaves(eqx.filter((b,sb),eqx.is_array))):
                np.testing.assert_array_equal(aa,bb)

    def test_native_crps_matches_pairwise_definition(self):
        samples=np.array([[0.,1.,3.],[2.,3.,4.],[1.,5.,2.]])
        truth=np.array([.5,2.,3.])
        brute=np.abs(samples-truth).mean(0)-.5*np.abs(samples[:,None]-samples[None,:]).mean((0,1))
        np.testing.assert_allclose(crps_array(samples,truth),brute)
        scores,pit=score_samples(samples,truth)
        self.assertAlmostEqual(scores["crps"],brute.mean())
        self.assertTrue(((pit>=0)&(pit<=1)).all())

    def test_validation_patience_respects_minimum_and_significant_improvements(self):
        req=dict(minimum_epochs=5,early_stopping_patience=3,early_stopping_min_delta=.01)
        rows=[dict(epoch=i+1,validation_loss=x) for i,x in enumerate([1.,.8,.801,.802,.799])]
        self.assertFalse(should_stop(rows[:4],req))
        self.assertTrue(should_stop(rows,req))
        rows[-1]["validation_loss"]=.78
        self.assertFalse(should_stop(rows,req))
        rows.extend(dict(epoch=i,validation_loss=.78) for i in (6,7,8))
        self.assertTrue(should_stop(rows,req))
        self.assertFalse(should_stop(rows,{**req,"early_stopping_patience":0}))

    def test_failed_building_invalidates_aggregate_instead_of_being_dropped(self):
        import pandas as pd
        scores=pd.DataFrame([
            dict(variant="a",policy="p",building=1,rmse=.1,crps=.05,simulation_failed=0),
            dict(variant="a",policy="p",building=2,rmse=None,crps=None,simulation_failed=1),
            dict(variant="b",policy="p",building=1,rmse=.2,crps=.1,simulation_failed=0),
            dict(variant="b",policy="p",building=2,rmse=.4,crps=.2,simulation_failed=0)])
        summary=aggregate_trace_scores(scores)
        self.assertTrue(np.isnan(summary.loc[("a","p"),"rmse"]))
        self.assertEqual(summary.loc[("a","p"),"failed_buildings"],1)
        self.assertEqual(summary.loc[("a","p"),"buildings"],2)
        self.assertAlmostEqual(summary.loc[("b","p"),"rmse"],.3)


if __name__=="__main__":unittest.main()
