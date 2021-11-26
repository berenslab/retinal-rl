import sys

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchinfo import summary
from captum.attr import NeuronGradient

from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations

from train_retina_rl import register_custom_components, custom_parse_args

def analyze(cfg):
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
    # env.seed(0)

    is_multiagent = is_multiagent_env(env)
    if not is_multiagent:
        env = MultiAgentWrapper(env)

    if hasattr(env.unwrapped, 'reset_on_init'):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)

    device = torch.device('cpu')
    #device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])
    enc = actor_critic.encoder.cnn_encoder
    obs = env.reset()

    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(device).float()
    #rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    #print(type(enc))
    obs0 = obs[0]
    obs1 = obs0['obs']
    #print(actor_critic(obs_torch,rnn_states))
    summary(enc,input_data=obs_torch['obs'])
    inp0 = obs_torch['obs']
    inp = inp0.numpy()[0,:,:,:]
    obssz = obs_torch['obs'].size()
    btchsz = list(obssz)
    btchsz[0] = 10000
    obsns = torch.normal(0,1,size=btchsz)
    outmtx = enc.nl2(enc.conv2(enc.nl1(enc.conv1(obsns))))
    print(outmtx.size())
    outs0 = outmtx[:,8,15,30].detach().numpy()
    outs1 = outmtx[:,10,15,30].detach().numpy()
    avg0 = np.zeros(btchsz[1:])
    avg1 = np.zeros(btchsz[1:])
    for i in range(0,btchsz[0]-1):
        avg0 = np.add(avg0, outs0[i] * obsns[i,:,:,:])
        avg1 = np.add(avg1, outs1[i] * obsns[i,:,:,:])

    fig, ((ax00,ax01),(ax10,ax11),(ax20,ax21)) = plt.subplots(3,2)
    plt00 = ax00.imshow(avg0[0,:,:])
    fig.colorbar(plt00, ax=ax00)
    plt10 = ax10.imshow(avg0[1,:,:])
    fig.colorbar(plt10, ax=ax10)
    plt20 = ax20.imshow(avg0[2,:,:])
    fig.colorbar(plt20, ax=ax20)
    plt01 = ax01.imshow(avg1[0,:,:])
    fig.colorbar(plt01, ax=ax01)
    plt11 = ax11.imshow(avg1[1,:,:])
    fig.colorbar(plt11, ax=ax11)
    plt21 = ax21.imshow(avg1[2,:,:])
    fig.colorbar(plt21, ax=ax21)
    plt.show()
    #print(enc(inp0))
    #print("HERE IS WHERE CAPTUM STARTS")
    #neuron_ig = NeuronGradient(enc.forward,enc.conv2)
    #grd0 = neuron_ig.attribute(inp0, (4,10,10))
    #grd = grd0.cpu().numpy()[0,:,:,:]
    #print(grd.shape)
    #print(inp.shape)
    #img = env.game.get_state().screen_buffer
    #print(img.shape)
    #print(grd)
    #print(env)
    #print(np.amax(inp))
    #print(np.amax(grd))
    #env.render()
    #print(grd)
    #env.viewer.imshow(np.transpose(inp, [1, 2, 0]))
    #time.sleep(2)
    #env.viewer.imshow(np.transpose(10000*grd, [1, 2, 0]))
    #time.sleep(2)
    #print(enc(obs_torch))
    #print(enc(obs_torch['obs']))

    # episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    # true_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    # num_frames = 0

    # last_render_start = time.time()

    # def max_frames_reached(frames):
    #     return max_num_frames is not None and frames > max_num_frames

    # obs = env.reset()
    # rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device)
    # episode_reward = np.zeros(env.num_agents)
    # finished_episode = [False] * env.num_agents

    # with torch.no_grad():
    #     while not max_frames_reached(num_frames):
    #         obs_torch = AttrDict(transform_dict_observations(obs))
    #         for key, x in obs_torch.items():
    #             obs_torch[key] = torch.from_numpy(x).to(device).float()

    #         policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)

    #         # sample actions from the distribution by default
    #         actions = policy_outputs.actions

    #         action_distribution = policy_outputs.action_distribution
    #         if isinstance(action_distribution, ContinuousActionDistribution):
    #             if not cfg.continuous_actions_sample:  # TODO: add similar option for discrete actions
    #                 actions = action_distribution.means

    #         actions = actions.cpu().numpy()

    #         rnn_states = policy_outputs.rnn_states

    #         for _ in range(render_action_repeat):
    #             if not cfg.no_render:
    #                 target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
    #                 current_delay = time.time() - last_render_start
    #                 time_wait = target_delay - current_delay

    #                 if time_wait > 0:
    #                     # log.info('Wait time %.3f', time_wait)
    #                     time.sleep(time_wait)

    #                 last_render_start = time.time()
    #                 env.render()

    #             obs, rew, done, infos = env.step(actions)

    #             episode_reward += rew
    #             num_frames += 1

    #             for agent_i, done_flag in enumerate(done):
    #                 if done_flag:
    #                     finished_episode[agent_i] = True
    #                     episode_rewards[agent_i].append(episode_reward[agent_i])
    #                     true_rewards[agent_i].append(infos[agent_i].get('true_reward', episode_reward[agent_i]))
    #                     log.info('Episode finished for agent %d at %d frames. Reward: %.3f, true_reward: %.3f', agent_i, num_frames, episode_reward[agent_i], true_rewards[agent_i][-1])
    #                     rnn_states[agent_i] = torch.zeros([get_hidden_size(cfg)], dtype=torch.float32, device=device)
    #                     episode_reward[agent_i] = 0

    #             # if episode terminated synchronously for all agents, pause a bit before starting a new one
    #             if all(done):
    #                 if not cfg.no_render:
    #                     env.render()
    #                 time.sleep(0.05)

    #             if all(finished_episode):
    #                 finished_episode = [False] * env.num_agents
    #                 avg_episode_rewards_str, avg_true_reward_str = '', ''
    #                 for agent_i in range(env.num_agents):
    #                     avg_rew = np.mean(episode_rewards[agent_i])
    #                     avg_true_rew = np.mean(true_rewards[agent_i])
    #                     if not np.isnan(avg_rew):
    #                         if avg_episode_rewards_str:
    #                             avg_episode_rewards_str += ', '
    #                         avg_episode_rewards_str += f'#{agent_i}: {avg_rew:.3f}'
    #                     if not np.isnan(avg_true_rew):
    #                         if avg_true_reward_str:
    #                             avg_true_reward_str += ', '
    #                         avg_true_reward_str += f'#{agent_i}: {avg_true_rew:.3f}'

    #                 log.info('Avg episode rewards: %s, true rewards: %s', avg_episode_rewards_str, avg_true_reward_str)
    #                 log.info('Avg episode reward: %.3f, avg true_reward: %.3f', np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]), np.mean([np.mean(true_rewards[i]) for i in range(env.num_agents)]))

    #             # VizDoom multiplayer stuff
    #             # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
    #             #     key = f'PLAYER{player}_FRAGCOUNT'
    #             #     if key in infos[0]:
    #             #         log.debug('Score for player %d: %r', player, infos[0][key])

    # env.close()

    # return ExperimentStatus.SUCCESS, np.mean(episode_rewards)


def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())
