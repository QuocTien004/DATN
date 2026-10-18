"""Regression contracts for opt-in Tien observation and evaluation features."""
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import torch

from envs.metadrive_wrapper import MetaDriveImageEnv
from evaluation.evaluate import LatentActorPolicy, evaluate_policy
from evaluation.safety_shield import AdaptiveTrafficAvoidanceShield, SafeLatentActorPolicy
from models.actor_critic import ActorCritic
from models.world_model import build_world_model


def vehicle(objects=(), speed=36.0):
    return SimpleNamespace(position=np.zeros(2), heading_theta=0.0, speed_km_h=speed,
                           lane_index=('0S0_0_', '1S0_0_', 0),
                           lidar=SimpleNamespace(get_surrounding_objects=lambda _: list(objects)))


def obstacle(x, y, speed=0.0, heading=0.0):
    return SimpleNamespace(position=np.array([x, y]), speed_km_h=speed, heading_theta=heading)


class TienEvalIntegrationTests(unittest.TestCase):
    def wrapper(self, cfg, ego=None):
        with patch.object(MetaDriveImageEnv, '_build_env'):
            env = MetaDriveImageEnv(cfg)
        env._env = SimpleNamespace(agent=ego)
        return env

    def policy(self, dim=23):
        torch.manual_seed(9)
        cfg = {'encoder': {'cnn_channels': [4], 'mlp_hidden': [4], 'embed_dim': 4},
               'rssm': {'deter_dim': 4, 'stoch_dim': 2, 'stoch_classes': 2, 'hidden_dim': 4},
               'predictors': {'reward_hidden': [4], 'continue_hidden': [4]}}
        wm = build_world_model(cfg, (16, 16, 3), dim, 2, torch.device('cpu'), include_decoder=False)
        ac = ActorCritic(dict(actor_hidden=[8], critic_hidden=[8], control_state_dim=dim,
                             control_state_source='observed', control_state_mean=[0.]*dim,
                             control_state_std=[1.]*dim), 4, 4, 2)
        return LatentActorPolicy(wm['encoder'], wm['rssm'], ac, device=torch.device('cpu'))

    def test_default_observation_preserves_state19_without_querying_lidar(self):
        env = self.wrapper({})
        state = np.arange(19, dtype=np.float32)
        with patch.object(env, '_extract_surrounding_features', side_effect=AssertionError):
            for obs in [state, {'image': np.zeros((16,16,3)), 'state': state}]:
                np.testing.assert_array_equal(env._format_obs(obs)['state'], state)

    def test_opt_in_observation23_preserves_prefix_and_normalizes_lead(self):
        env = self.wrapper({'state_dim':23}, vehicle([obstacle(10., 0.)]))
        state = np.arange(19, dtype=np.float32)
        value = env._format_obs(state)['state']
        np.testing.assert_array_equal(value[:19], state)
        np.testing.assert_allclose(value[19:], [.2, -.5, 0., 0.], atol=1e-7)
        np.testing.assert_array_equal(env._format_obs(value)['state'], value)

    def test_opt_in_features_use_ego_frame(self):
        ego = vehicle([obstacle(0., 10., heading=np.pi/2)])
        ego.heading_theta = np.pi/2
        env = self.wrapper({'state_dim':23}, ego)
        np.testing.assert_allclose(env._extract_surrounding_features(), [.2,-.5,0,0], atol=1e-7)

    def test_state_dimension_mismatch_fails_early(self):
        for dim, observed in [(23,18),(19,23)]:
            with self.assertRaisesRegex(ValueError,'state_dim'):
                self.wrapper({'state_dim':dim})._format_obs(np.zeros(observed))

    def test_observed23_policy_reads_current_state_and_keeps_rl_locked(self):
        policy = self.policy()
        obs = dict(image=np.zeros((16,16,3),dtype=np.uint8),state=np.arange(23,dtype=np.float32)/23)
        with patch.object(policy.actor_critic.actor,'forward',wraps=policy.actor_critic.actor.forward) as forward:
            a = policy(obs)
            torch.testing.assert_close(forward.call_args.kwargs['control_state'],torch.tensor(obs['state'])[None])
        self.assertTrue(np.isfinite(a).all())
        self.assertIsNone(policy.rssm.continuous)
        with self.assertRaisesRegex(ValueError,'no validated'):
            policy.actor_critic.require_imagination_ready()

    def test_shield_brakes_for_lead_but_ignores_opposing_adjacent_vehicle(self):
        shield = AdaptiveTrafficAvoidanceShield()
        action = np.array([0.,.8],dtype=np.float32)
        result,diag = shield.modulate_action(action,vehicle([obstacle(3,0)]))
        self.assertEqual(result[1],-1.)
        self.assertEqual(diag['throttle_override'],1.)
        result,_ = shield.modulate_action(action,vehicle([obstacle(3,3,heading=np.pi)]))
        np.testing.assert_array_equal(result,action)
        np.testing.assert_array_equal(action,[0.,np.float32(.8)])

    def test_wrapper_records_executed_action_in_next_posterior(self):
        raw=self.policy()
        safe=SafeLatentActorPolicy(raw,env=SimpleNamespace(agent=vehicle([obstacle(3,0)])))
        obs=dict(image=np.zeros((16,16,3),dtype=np.uint8),state=np.zeros(23,dtype=np.float32))
        first=safe(obs)
        self.assertEqual(first[1],-1.)
        with patch.object(raw.rssm,'observe_step',wraps=raw.rssm.observe_step) as observe:
            safe(obs)
            torch.testing.assert_close(observe.call_args.args[1],torch.tensor(first)[None])
        safe.shield.stuck_counter=9
        safe.reset()
        self.assertEqual(safe.shield.stuck_counter,0)
        torch.testing.assert_close(raw._previous_action,torch.zeros(1,2))

    def test_all_interventions_reported_and_actions_are_not_aliased(self):
        shield=AdaptiveTrafficAvoidanceShield()
        a=np.array([.8,.9],dtype=np.float32)
        turn,diag=shield.modulate_action(a,vehicle())
        self.assertLess(turn[1],a[1]);self.assertEqual(diag['throttle_override'],1.)
        a=np.array([.8,-.3],dtype=np.float32)
        cruise,diag=shield.modulate_action(a,vehicle(speed=0))
        self.assertNotEqual(cruise[0],a[0]);self.assertEqual(diag['steering_override'],1.)
        raw=self.policy();raw.record_executed_action(cruise);cruise[:]=0
        self.assertNotEqual(float(raw._previous_action.abs().sum()),0.)

    def test_reject_invalid_actions_and_shield_settings(self):
        shield=AdaptiveTrafficAvoidanceShield()
        for a in [np.zeros(3),np.array([np.nan,0]),np.array([2.,0])]:
            with self.assertRaises(ValueError):shield.modulate_action(a,None)
        for kwargs in [dict(threat_distance=0),dict(min_gap=float('nan')),dict(max_decel=1)]:
            with self.assertRaises(ValueError):AdaptiveTrafficAvoidanceShield(**kwargs)

    def test_eval_labels_shield_counts_and_preserves_real_return(self):
        class Env:
            env_cfg={'horizon':2}
            agent=vehicle([obstacle(3,0)])
            def reset(self,seed=None):self.n=0;return {},{}
            def step(self,a):
                self.n+=1
                return {},-2.,self.n==2,False,{'route_completion':.1,'crash_vehicle':self.n==2}
        with tempfile.TemporaryDirectory() as td:
            path=Path(td)/'eval.json'
            metrics=evaluate_policy(Env(),lambda _:np.array([0,.8]),num_episodes=2,
                                    output_path=path,shield=AdaptiveTrafficAvoidanceShield())
            result=json.loads(path.read_text())
            self.assertTrue(result['safety_shield'])
            self.assertEqual([r['shield_interventions'] for r in result['episodes']],[2,2])
            self.assertEqual(metrics['mean_episode_return'],-4.)


if __name__=='__main__':unittest.main()
