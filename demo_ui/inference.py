"""Read-only checkpoint adapter. All tensor work stays in the worker process."""
from __future__ import annotations
import copy
from pathlib import Path

import numpy as np
import torch

from evaluation.evaluate import LatentActorPolicy
from models.actor_critic import ActorCritic
from models.predictors import predict_from_state
from models.world_model import build_world_model, load_world_model_state, validate_world_model_metadata
from demo_ui.recording import sha256
from utils.config import load_experiment_configs


def validate_observation(image, state, image_shape, state_dim):
    image = np.asarray(image)
    vector = np.asarray(state, dtype=np.float32)
    if image.shape != tuple(image_shape) or image.dtype != np.uint8:
        raise ValueError(f'Ảnh phải là RGB uint8 kích thước {tuple(image_shape)}, nhận {image.shape}/{image.dtype}. Không tạo ảnh thay thế.')
    if vector.shape != (state_dim,) or not np.isfinite(vector).all():
        raise ValueError(f'Cần đúng {state_dim} cảm biến hữu hạn theo đúng thứ tự của checkpoint.')
    return {'image': image, 'state': vector}


def _summary(tensor):
    array = tensor.detach().float().cpu().numpy().reshape(-1)
    return dict(dim=int(array.size), mean=float(array.mean()), std=float(array.std()),
                minimum=float(array.min()), maximum=float(array.max()),
                preview=array[:32].tolist())


class InferenceEngine:
    def __init__(self, checkpoint, wm_checkpoint=None, device='cpu', config='configs/train.yaml'):
        if device == 'cuda' and not torch.cuda.is_available():
            raise ValueError('CUDA chưa khả dụng. Chọn CPU hoặc kiểm tra cài đặt PyTorch.')
        self.device = torch.device(device)
        # Only local, trusted project checkpoints. Replay never unpickles data.
        agent = torch.load(checkpoint, map_location='cpu', weights_only=False)
        configs = load_experiment_configs(config)
        if 'models' in agent:
            if wm_checkpoint:
                raise ValueError('Checkpoint đã chứa WM. Để trống đường dẫn WM riêng.')
            wm = agent
            wm_source = checkpoint
        else:
            wm_source = wm_checkpoint or agent.get('world_model_checkpoint')
            if not wm_source or not Path(wm_source).is_file():
                raise ValueError('Checkpoint chỉ chứa Actor. Cần chọn đúng checkpoint WM đi kèm.')
            wm = torch.load(wm_source, map_location='cpu', weights_only=False)
        self.image_shape = tuple(wm.get('image_shape', agent.get('image_shape', (64,64,3))))
        self.state_dim = int(wm.get('state_dim', agent.get('state_dim', 19)))
        action_dim = int(wm.get('action_dim', agent.get('action_dim', 2)))
        if action_dim != 2:
            raise ValueError('Demo MetaDrive cần action_dim=2.')
        validate_world_model_metadata(wm, image_shape=self.image_shape,
                                      state_dim=self.state_dim, action_dim=2)
        validate_world_model_metadata(agent, image_shape=self.image_shape,
                                      state_dim=self.state_dim, action_dim=2)
        self.env_cfg = copy.deepcopy(agent.get('env_cfg', configs['env']))
        self.env_cfg.update(state_dim=self.state_dim, image_height=self.image_shape[0],
                            image_width=self.image_shape[1], image_observation=True,
                            image_source='rgb_camera')
        self.models = build_world_model(wm.get('wm_cfg', configs['world_model']), self.image_shape,
                                        self.state_dim, 2, self.device, include_decoder=False)
        load_world_model_state(self.models, wm)
        ac_cfg = agent.get('actor_critic_cfg')
        if ac_cfg is None:
            ac_cfg = dict(configs['train'].get('actor_critic', {}), mean_transform='clamp',
                          std_transform='log_clamp', min_std=.30, max_std=1.)
        if agent.get('control_input') in {'continuous', 'predicted'}:
            ac_cfg = dict(ac_cfg, control_state_source=agent['control_input'])
        rssm = self.models['rssm']
        ac = ActorCritic(ac_cfg, rssm.deter_dim, rssm.stoch_size, 2).to(self.device)
        if ac.actor.control_state_source=='observed' and ac.actor.control_state_dim not in (0,self.state_dim):
            raise ValueError('Actor observed-state không khớp chiều observation của WM.')
        ac.load_state_dict(agent['actor_critic'], strict=True)
        self.policy = LatentActorPolicy(self.models['encoder'], rssm, ac, device=self.device,
                                       deterministic=True,
                                       control_predictor=self.models.get('control_state')
                                       if ac.actor.control_state_source == 'predicted' else None)
        for model in self.models.values():
            model.eval().requires_grad_(False)
        ac.eval().requires_grad_(False)
        self.embedding = None
        self.models['encoder'].register_forward_hook(self._capture_embedding)
        try:
            ac.require_imagination_ready()
            structural_ready = True
        except ValueError:
            structural_ready = False
        self.metadata = dict(checkpoint_name=Path(checkpoint).name, checkpoint_sha256=sha256(checkpoint),
                             wm_name=Path(wm_source).name, wm_sha256=sha256(wm_source),
                             image_shape=list(self.image_shape), state_dim=self.state_dim,
                             action_dim=2, control_source=ac.actor.control_state_source,
                             continuous_branch=rssm.continuous is not None, device=str(self.device),
                             actor_c23_rules=ac.actor.control_state_dim >= 23,
                             structural_imagination_ready=structural_ready, rl_locked=True,
                             actor_critic_cfg=ac_cfg, wm_cfg=wm.get('wm_cfg'),
                             torch_version=torch.__version__)
        self.steps = 0
        self.reset()

    def _capture_embedding(self, module, inputs, output):
        self.embedding = output.detach()

    def reset(self):
        self.policy.reset()
        self.steps = 0

    def observe(self, image, state, *, previous_action=None, reset=False):
        observation = validate_observation(image, state, self.image_shape, self.state_dim)
        if reset:
            self.reset()
        if self.steps and previous_action is None:
            raise ValueError('Nối lịch sử RSSM cần action thực thi ở bước trước.')
        if previous_action is not None:
            if not self.steps and np.any(np.asarray(previous_action) != 0):
                raise ValueError('Bước đầu episode dùng action trước bằng 0. Reset khi bắt đầu episode mới.')
            self.policy.record_executed_action(previous_action)
        with torch.inference_mode():
            action = self.policy(observation)
            latent = self.policy._latent
            summaries = {name: _summary(value) for name, value in latent.items() if name in {'h','z','c','control'}}
            summaries['embedding'] = _summary(self.embedding)
            heads = self.heads(latent)
        self.steps += 1
        return dict(action=action.tolist(), features=summaries, posterior=heads,
                    history_steps=self.steps, state=observation['state'].tolist())

    def heads(self, latent):
        with torch.inference_mode():
            sym_reward = float(predict_from_state(self.models['reward'], latent).reshape(-1)[0])
            cont = float(torch.sigmoid(predict_from_state(self.models['continue'], latent)).reshape(-1)[0])
        # Same display ceiling as imagination. Keep un-clipped symlog output too.
        reward = float(np.sign(sym_reward)*np.expm1(min(abs(sym_reward), 3.26)))
        return dict(reward_symlog=sym_reward, reward=reward, continue_probability=cont,
                    reward_display_clipped=abs(sym_reward)>3.26)

    def prior(self, executed_action):
        action = torch.as_tensor(executed_action, device=self.device, dtype=torch.float32).reshape(1,2)
        with torch.inference_mode():
            state, _ = self.models['rssm'].imagine_step(self.policy._latent, action, deterministic=True)
            return self.heads(state)
