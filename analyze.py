# System
import sys

# Numerics
import torch
import numpy as np

#from torchinfo import summary
#from captum.attr import NeuronGradient

# Sample Factory
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.utils.action_distributions import ContinuousActionDistribution
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.utils.arguments import load_from_checkpoint
from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper
from retina_rl.encoders import register_custom_encoders
from retina_rl.environment import custom_parse_args,register_custom_doom_environments

# retina-rl
from retina_rl.visualization import *


def analyze(cfg, max_num_frames=1e3):

    ### Loading Model

    cfg = load_from_checkpoint(cfg)

    render_action_repeat = cfg.render_action_repeat if cfg.render_action_repeat is not None else cfg.env_frameskip
    if render_action_repeat is None:
        log.warning('Not using action repeat!')
        render_action_repeat = 1
    log.debug('Using action repeat %d during evaluation', render_action_repeat)

    cfg.env_frameskip = 1  # for evaluation
    cfg.num_envs = 1

    def make_env_func(env_config):
        return create_env(cfg.env, cfg=cfg, env_config=env_config)

    env = make_env_func(AttrDict({'worker_index': 0, 'vector_index': 0}))
    env = MultiAgentWrapper(env)
    # env.seed(0)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])

    num_frames = 0

    def max_frames_reached(frames):
        return max_num_frames is not None and frames > max_num_frames

    ### Parsing out the encoder and preparing some analyses

    enc = actor_critic.encoder.cnn_encoder

    obs = env.reset()
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(device).float()

    isz = list(obs_torch['obs'].size())[1:]
    outmtx = enc.nl(enc.conv2(enc.nl(enc.conv1(obs_torch['obs']))))
    osz = list(outmtx.size())[1:]

    ### Running and saving a simulation

    img0 = obs_torch['obs'][0,:,:,:]
    imgs = [np.transpose(img0.cpu().detach().numpy(),(1,2,0)).astype(np.uint8)]

    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)

    with torch.no_grad():

        while not max_frames_reached(num_frames):

            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

            # sample actions from the distribution by default
            actions = policy_outputs.actions

            action_distribution = policy_outputs.action_distribution
            if isinstance(action_distribution, ContinuousActionDistribution):
                if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
                    actions = action_distribution.means

            actions = actions.cpu().numpy()

            rnn_states = policy_outputs.rnn_states

            for _ in range(render_action_repeat):

                obs,_,_,_ = env.step(actions)
                obs_torch = AttrDict(transform_dict_observations(obs))
                for key, x in obs_torch.items():
                    obs_torch[key] = torch.from_numpy(x).to(device).float()

                img0 = obs_torch['obs'][0,:,:,:]
                img = np.transpose(img0.cpu().detach().numpy(),(1,2,0)).astype(np.uint8)
                imgs.append(img)


                num_frames += 1

    env.close()

    # Printing encoder stats
    print(enc)

    # Analysing encoder
    print("Input Size: ", isz)
    print("Output Size: ", osz)

    save_simulation_gif(cfg,imgs)
    save_receptive_fields_plot(cfg,device,enc,isz,osz)
    ### Analysis and plotting

def main():
    """Script entry point."""
    register_custom_encoders()
    register_custom_doom_environments()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())
