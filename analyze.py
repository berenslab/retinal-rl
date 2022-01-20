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

from retinal_rl.environment import register_retinal_environment
from retinal_rl.parameters import custom_parse_args
from retinal_rl.encoders import register_encoders

# retina-rl
from retinal_rl.visualization import *


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

    enc = actor_critic.encoder.base_encoder

    obs = env.reset()
    print("\nEnvironment Info:")
    print(env.unwrapped.get_info())
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(device).float()

    ### Analysing encoder
    print("\nActor-Critic Stats:")
    # Printing encoder stats
    print(enc)
    # logging comp graph to tensorboard
    tb_path = cfg.train_dir +  "/" + cfg.experiment + '/.summary/0/'
    writer = SummaryWriter(tb_path)
    writer.add_graph(enc, obs_torch['obs'])
    writer.close()
    print(actor_critic)

    n_conv_lay = len(enc.conv_head)//2 # sequential list
    for lay in range(1, n_conv_lay+1):
        save_receptive_fields_plot(cfg,device,enc,lay,obs_torch)

    ### Running and saving a simulation

    img0 = obs_torch['obs'][0,:,:,:]
    imgs = [np.transpose(img0.cpu().detach().numpy(),(1,2,0)).astype(np.uint8)]
    conv_acts = [ [] for _ in range(n_conv_lay) ]
    fc_acts = []
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

                img0 = obs_torch['obs'][0,:,:,:] # activation of input layer (=pixels)
                img = np.transpose(img0.cpu().detach().numpy(),(1,2,0)).astype(np.uint8)
                imgs.append(img)

                for lay in range(1,n_conv_lay+1):
                    conv_act_torch = enc.conv_head[0:(lay*2)](obs_torch['obs'])[0,:,:,:].permute(1, 2, 0) # activation of intermediate conv layers
                    conv_act_np = conv_act_torch.cpu().detach().numpy()
                    conv_acts[lay-1].append(conv_act_np)

                fc_act_torch = enc.forward(obs_torch['obs']) # activation of output fc layer
                fc_act_np = fc_act_torch.cpu().detach().numpy()
                fc_acts.append(fc_act_np)
                num_frames += 1

    env.close()

    # input layer
    save_simulation_gif(cfg,imgs)

    # intermediate conv layers
    if cfg.analyze_acts:
        for lay in range(n_conv_lay):
            save_activations_gif(cfg, imgs, conv_acts, lay)

    #TODO: add analysis of output fc layer

    ### Analysis and plotting

def main():
    """Script entry point."""
    register_retinal_environment()
    register_encoders()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())
