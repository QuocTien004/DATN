"""Checkpoint compatibility and recurrent inference timing, with tiny unit fixtures."""
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from demo_ui.inference import InferenceEngine, validate_observation
from demo_ui.worker import ModelWorker
from models.world_model import build_world_model
from models.actor_critic import ActorCritic


def checkpoint(path):
    torch.manual_seed(17)
    cfg={'encoder':{'cnn_channels':[4],'mlp_hidden':[4],'embed_dim':4},
         'rssm':{'deter_dim':4,'stoch_dim':2,'stoch_classes':2,'hidden_dim':4},
         'predictors':{'reward_hidden':[4],'continue_hidden':[4]}}
    models=build_world_model(cfg,(16,16,3),23,2,torch.device('cpu'),include_decoder=False)
    ac_cfg=dict(actor_hidden=[8],critic_hidden=[8],control_state_dim=23,
                control_state_source='observed',control_state_mean=[0.]*23,control_state_std=[1.]*23)
    ac=ActorCritic(ac_cfg,4,4,2)
    torch.save(dict(models={k:v.state_dict() for k,v in models.items()},actor_critic=ac.state_dict(),
                    actor_critic_cfg=ac_cfg,wm_cfg=cfg,image_shape=(16,16,3),state_dim=23,action_dim=2,
                    transition_contract_version=2),path)


class InferenceTests(unittest.TestCase):
    def test_strict_model_input_rejects_shape_scale_and_nonfinite_sensors(self):
        rgb=np.zeros((16,16,3),dtype=np.uint8)
        for image,state in [(rgb,[0.]*19),(rgb,np.full(23,np.nan)),
                            (rgb.astype(float),[0.]*23),(rgb[:8],[0.]*23)]:
            with self.assertRaises(ValueError):validate_observation(image,state,(16,16,3),23)
        self.assertEqual(validate_observation(rgb,[0.]*23,(16,16,3),23)['state'].shape,(23,))

    def test_history_requires_executed_action_and_reset_reproduces_initial_action(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'policy.pt';checkpoint(path)
            engine=InferenceEngine(path)
            rgb=np.zeros((16,16,3),dtype=np.uint8);state=np.zeros(23,dtype=np.float32)
            first=engine.observe(rgb,state,reset=True)
            with self.assertRaisesRegex(ValueError,'action thực thi'):engine.observe(rgb,state)
            second=engine.observe(rgb,state,previous_action=[.3,-1.])
            self.assertEqual(second['history_steps'],2)
            reset=engine.observe(rgb,state,reset=True)
            np.testing.assert_array_equal(reset['action'],first['action'])
            self.assertEqual(reset['history_steps'],1)
            self.assertFalse(engine.metadata['continuous_branch'])
            self.assertTrue(engine.metadata['rl_locked'])

    def test_prior_diagnostics_do_not_advance_posterior_or_change_weights(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'policy.pt';checkpoint(path)
            engine=InferenceEngine(path)
            engine.observe(np.zeros((16,16,3),dtype=np.uint8),np.zeros(23,dtype=np.float32))
            before={k:v.clone() for k,v in engine.policy._latent.items()}
            parameters={k:v.clone() for k,v in engine.models['rssm'].state_dict().items()}
            heads=engine.prior([.4,-.5])
            self.assertTrue(0<=heads['continue_probability']<=1)
            self.assertTrue(np.isfinite(heads['reward']))
            for k,v in before.items():torch.testing.assert_close(engine.policy._latent[k],v)
            for k,v in parameters.items():torch.testing.assert_close(engine.models['rssm'].state_dict()[k],v)
            self.assertTrue(all(not p.requires_grad for m in engine.models.values() for p in m.parameters()))

    def test_spawned_worker_survives_bad_input_and_closes_cleanly(self):
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'policy.pt';checkpoint(path)
            worker=ModelWorker(timeout=60)
            try:
                self.assertEqual(worker.call('load',checkpoint=str(path))['state_dim'],23)
                with self.assertRaises(ValueError):
                    worker.call('infer',image=np.zeros((8,8,3),dtype=np.uint8),state=[0.]*23,reset=True)
                result=worker.call('infer',image=np.zeros((16,16,3),dtype=np.uint8),state=[0.]*23,reset=True)
                self.assertEqual(result['history_steps'],1)
            finally:worker.close()
            self.assertFalse(worker.process.is_alive())


if __name__=='__main__':unittest.main()
