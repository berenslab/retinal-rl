### Util for preparing simulations and data for analysis

import time
from collections import deque
from typing import Dict, Tuple

import gym
import numpy as np
import torch
from torch import Tensor

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.algo.utils.make_env import make_env_func_batched,BatchedVecEnv
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.algo.utils.rl_utils import make_dones, prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.huggingface.huggingface_utils import generate_model_card, generate_replay_video, push_to_hf
from sample_factory.model.actor_critic import create_actor_critic,ActorCritic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config, StatusCode
from sample_factory.utils.utils import debug_log_every_n, experiment_dir, log

def visualize_policy_inputs(normalized_obs: Dict[str, Tensor]) -> None:
    """
    Display actual policy inputs after all wrappers and normalizations using OpenCV imshow.
    """
    import cv2

    if "obs" not in normalized_obs.keys():
        return

    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]
    if obs.dim() != 3:
        # this function is only for RGB images
        return

    # convert to HWC
    obs = obs.permute(1, 2, 0)
    # convert to numpy
    obs = obs.cpu().numpy()
    # convert to uint8
    obs = cv2.normalize(
        obs, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8UC1
    )  # this will be different frame-by-frame but probably good enough to give us an idea?
    scale = 5
    obs = cv2.resize(obs, (obs.shape[1] * scale, obs.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cv2.imshow("policy inputs", obs)
    cv2.waitKey(delay=1)

def render_frame(cfg, env, video_frames, num_episodes, last_render_start) -> float:
    render_start = time.time()

    if cfg.save_video:
        need_video_frame = len(video_frames) < cfg.video_frames or cfg.video_frames < 0 and num_episodes == 0
        if need_video_frame:
            frame = env.render()
            if frame is not None:
                video_frames.append(frame.copy())
    else:
        if not cfg.no_render:
            target_delay = 1.0 / cfg.fps if cfg.fps > 0 else 0
            current_delay = render_start - last_render_start
            time_wait = target_delay - current_delay

            if time_wait > 0:
                # log.info("Wait time %.3f", time_wait)
                time.sleep(time_wait)

            try:
                env.render()
            except (gym.error.Error, TypeError) as ex:
                debug_log_every_n(1000, f"Exception when calling env.render() {str(ex)}")

    return render_start

def get_ac_env(cfg: Config) -> Tuple[ActorCritic, BatchedVecEnv]:
    """
    Load the model from checkpoint, initialize the environment, and return both.
    """
    #verbose = False

    cfg = load_from_checkpoint(cfg)

    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
    assert (
        cfg.env_frameskip % eval_env_frameskip == 0
    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")

    cfg.num_envs = 1

    render_mode = "human"
    if cfg.save_video:
        render_mode = "rgb_array"
    elif cfg.no_render:
        render_mode = None

    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )
    #env_info = extract_env_info(env, cfg)

    if hasattr(env.unwrapped, "reset_on_init"):
        # reset call ruins the demo recording for VizDoom
        env.unwrapped.reset_on_init = False

    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
    actor_critic.eval()

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict["model"])

    return actor_critic,env

def write_onxx(ac : ActorCritic, env : BatchedVecEnv) -> None:
    """
    Write an onxx file of the saved model.
    """

    obs, _ = env.reset()
    normalized_obs = prepare_and_normalize_obs(ac, obs)
    enc = ac.encoder.basic_encoder
    obs = normalized_obs["obs"]
    # visualize obs only for the 1st agent
    obs = obs[0]

    # Note that onnx can't process dictionary inputs
    torch.onnx.export(enc,obs,"actor-critic.onnx",verbose=False,input_names=["observation"],output_names=["latent_state"])

    #episode_rewards = [deque([], maxlen=100) for _ in range(env.num_agents)]
    #true_objectives = [deque([], maxlen=100) for _ in range(env.num_agents)]
    #num_frames = 0

    #last_render_start = time.time()

    #def max_frames_reached(frames):
    #    return cfg.max_num_frames is not None and frames > cfg.max_num_frames

    #reward_list = []

    #obs, infos = env.reset()
    #rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    #episode_reward = None
    #finished_episode = [False for _ in range(env.num_agents)]

    #video_frames = []
    #num_episodes = 0

    #with torch.no_grad():
    #    while not max_frames_reached(num_frames):
    #        normalized_obs = prepare_and_normalize_obs(actor_critic, obs)

    #        if not cfg.no_render:
    #            visualize_policy_inputs(normalized_obs)
    #        policy_outputs = actor_critic(normalized_obs, rnn_states)

    #        # sample actions from the distribution by default
    #        actions = policy_outputs["actions"]

    #        if cfg.eval_deterministic:
    #            action_distribution = actor_critic.action_distribution()
    #            actions = argmax_actions(action_distribution)

    #        # actions shape should be [num_agents, num_actions] even if it's [1, 1]
    #        if actions.ndim == 1:
    #            actions = unsqueeze_tensor(actions, dim=-1)
    #        actions = preprocess_actions(env_info, actions)

    #        rnn_states = policy_outputs["new_rnn_states"]

    #        for _ in range(render_action_repeat):
    #            last_render_start = render_frame(cfg, env, video_frames, num_episodes, last_render_start)

    #            obs, rew, terminated, truncated, infos = env.step(actions)
    #            dones = make_dones(terminated, truncated)
    #            infos = [{} for _ in range(env_info.num_agents)] if infos is None else infos

    #            if episode_reward is None:
    #                episode_reward = rew.float().clone()
    #            else:
    #                episode_reward += rew.float()

    #            num_frames += 1
    #            if num_frames % 100 == 0:
    #                log.debug(f"Num frames {num_frames}...")

    #            dones = dones.cpu().numpy()
    #            for agent_i, done_flag in enumerate(dones):
    #                if done_flag:
    #                    finished_episode[agent_i] = True
    #                    rew = episode_reward[agent_i].item()
    #                    episode_rewards[agent_i].append(rew)

    #                    true_objective = rew
    #                    if isinstance(infos, (list, tuple)):
    #                        true_objective = infos[agent_i].get("true_objective", rew)
    #                    true_objectives[agent_i].append(true_objective)

    #                    if verbose:
    #                        log.info(
    #                            "Episode finished for agent %d at %d frames. Reward: %.3f, true_objective: %.3f",
    #                            agent_i,
    #                            num_frames,
    #                            episode_reward[agent_i],
    #                            true_objectives[agent_i][-1],
    #                        )
    #                    rnn_states[agent_i] = torch.zeros([get_rnn_size(cfg)], dtype=torch.float32, device=device)
    #                    episode_reward[agent_i] = 0

    #                    if cfg.use_record_episode_statistics:
    #                        # we want the scores from the full episode not a single agent death (due to EpisodicLifeEnv wrapper)
    #                        if "episode" in infos[agent_i].keys():
    #                            num_episodes += 1
    #                            reward_list.append(infos[agent_i]["episode"]["r"])
    #                    else:
    #                        num_episodes += 1
    #                        reward_list.append(true_objective)

    #            # if episode terminated synchronously for all agents, pause a bit before starting a new one
    #            if all(dones):
    #                render_frame(cfg, env, video_frames, num_episodes, last_render_start)
    #                time.sleep(0.05)

    #            if all(finished_episode):
    #                finished_episode = [False] * env.num_agents
    #                avg_episode_rewards_str, avg_true_objective_str = "", ""
    #                for agent_i in range(env.num_agents):
    #                    avg_rew = np.mean(episode_rewards[agent_i])
    #                    avg_true_obj = np.mean(true_objectives[agent_i])

    #                    if not np.isnan(avg_rew):
    #                        if avg_episode_rewards_str:
    #                            avg_episode_rewards_str += ", "
    #                        avg_episode_rewards_str += f"#{agent_i}: {avg_rew:.3f}"
    #                    if not np.isnan(avg_true_obj):
    #                        if avg_true_objective_str:
    #                            avg_true_objective_str += ", "
    #                        avg_true_objective_str += f"#{agent_i}: {avg_true_obj:.3f}"

    #                log.info(
    #                    "Avg episode rewards: %s, true rewards: %s", avg_episode_rewards_str, avg_true_objective_str
    #                )
    #                log.info(
    #                    "Avg episode reward: %.3f, avg true_objective: %.3f",
    #                    np.mean([np.mean(episode_rewards[i]) for i in range(env.num_agents)]),
    #                    np.mean([np.mean(true_objectives[i]) for i in range(env.num_agents)]),
    #                )

    #            # VizDoom multiplayer stuff
    #            # for player in [1, 2, 3, 4, 5, 6, 7, 8]:
    #            #     key = f'PLAYER{player}_FRAGCOUNT'
    #            #     if key in infos[0]:
    #            #         log.debug('Score for player %d: %r', player, infos[0][key])

    #        if num_episodes >= cfg.max_num_episodes:
    #            break

    #env.close()

    #if cfg.save_video:
    #    if cfg.fps > 0:
    #        fps = cfg.fps
    #    else:
    #        fps = 30
    #    generate_replay_video(experiment_dir(cfg=cfg), video_frames, fps)

    #if cfg.push_to_hub:
    #    generate_model_card(
    #        experiment_dir(cfg=cfg),
    #        cfg.algo,
    #        cfg.env,
    #        cfg.hf_repository,
    #        reward_list,
    #        cfg.enjoy_script,
    #        cfg.train_script,
    #    )
    #    push_to_hf(experiment_dir(cfg=cfg), cfg.hf_repository)

    #return ExperimentStatus.SUCCESS, sum([sum(episode_rewards[i]) for i in range(env.num_agents)]) / sum(
    #    [len(episode_rewards[i]) for i in range(env.num_agents)]
    #)

# getting environment and actor critic from checkpoint

#def get_env_ac(cfg):
#
#    cfg = load_from_checkpoint(cfg)
#
#    eval_env_frameskip: int = cfg.env_frameskip if cfg.eval_env_frameskip is None else cfg.eval_env_frameskip
#    assert (
#        cfg.env_frameskip % eval_env_frameskip == 0
#    ), f"{cfg.env_frameskip=} must be divisible by {eval_env_frameskip=}"
#    render_action_repeat: int = cfg.env_frameskip // eval_env_frameskip
#    cfg.env_frameskip = cfg.eval_env_frameskip = eval_env_frameskip
#    log.debug(f"Using frameskip {cfg.env_frameskip} and {render_action_repeat=} for evaluation")
#
#    cfg.num_envs = 1
#
#    render_mode = "human"
#    if cfg.save_video:
#        render_mode = "rgb_array"
#    elif cfg.no_render:
#        render_mode = None
#
#    env = make_env_func_batched(
#        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
#    )
#    #env_info = extract_env_info(env, cfg)
#
#    if hasattr(env.unwrapped, "reset_on_init"):
#        # reset call ruins the demo recording for VizDoom
#        env.unwrapped.reset_on_init = False
#
#    actor_critic = create_actor_critic(cfg, env.observation_space, env.action_space)
#
#    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
#    actor_critic.model_to_device(device)
#
#    policy_id = cfg.policy_index
#    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id))
#    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)
#    actor_critic.load_state_dict(checkpoint_dict['model'])
#
#    return env, actor_critic

# Running experience with the extracted agent and exporting interaction

#def simulate(cfg, env, actor_critic):
#    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
#    n_conv_lay = cfg.vvs_depth + 2 # vvs + input + bottelneck layer
#
#    obs = env.reset() # this is the first observation when agent is spawned
#    obs_torch = AttrDict(transform_dict_observations(obs))
#    for key, x in obs_torch.items():
#        obs_torch[key] = torch.from_numpy(x).to(device).float()
#
#    # initialising rnn state and observation
#    obs = env.reset()
#    rnn_states = torch.zeros([env.num_agents, get_hidden_size(cfg)], dtype=torch.float32, device=device) # this is how to initialize
#
#    # initialise matrices for saving outputs
#    t_max = int(cfg.analyze_max_num_frames)
#    all_img = np.zeros((cfg.res_h, cfg.res_w, 3, t_max)).astype(np.uint8)
#    all_fc_act = np.zeros((cfg.hidden_size, t_max))
#    all_rnn_act = np.zeros((cfg.hidden_size, t_max))
#    all_v_act = np.zeros(t_max)
#    all_actions = np.zeros((2, t_max))
#    all_health = np.zeros(t_max)
#    conv_acts = [ [] for _ in range(n_conv_lay) ]
#
#    num_frames = 0 # counter
#
#    with torch.no_grad():
#        while t_max>num_frames:
#
#            policy_outputs = actor_critic(obs_torch, rnn_states, with_action_distribution=True)
#            actions = policy_outputs.actions
#            rnn_states = policy_outputs.rnn_states
#            actions = actions.cpu().numpy() # to feed to vizdoom (CPU based)
#
#            # here only valid for no action repeat
#            obs = env.step(actions)[0]
#            obs_torch = AttrDict(transform_dict_observations(obs))
#            for key, x in obs_torch.items():
#                obs_torch[key] = torch.from_numpy(x).to(device).float()
#
#            # extracting all activations to save
#            img = obs_to_img(obs).astype(np.uint8) # to save later
#            fc_act = actor_critic.encoder.base_encoder.forward(obs_torch['obs']).cpu().detach().numpy()
#            rnn_act = rnn_states.cpu().detach().numpy()
#            v_act = actor_critic.critic_linear(rnn_states).cpu().detach().numpy()
#            actions = np.array(actions)
#            health = env.unwrapped.get_info()['HEALTH'] # environment info (health etc.)
#
#            enc = actor_critic.encoder.base_encoder
#            for lay in range(n_conv_lay):
#                    conv_act_torch = enc.conv_head[0:((lay+1)*2)](obs_torch['obs'])[0,:,:,:].permute(1, 2, 0)
#                    conv_act_np = conv_act_torch.cpu().detach().numpy()
#                    conv_acts[lay].append(conv_act_np)
#
#            # saving
#            all_img[:,:,:,num_frames] = img
#            all_fc_act[:,num_frames] = fc_act
#            all_rnn_act[:,num_frames] = rnn_act
#            all_v_act[num_frames] = v_act
#            all_actions[:,num_frames] = actions
#            all_health[num_frames] = health
#            # conv acts is a list
#
#            num_frames+=1
#            if num_frames % int(t_max/10) == 0:
#                print(f'Simulated {num_frames} / {int(t_max)} environment steps.')
#
#    analyze_out = {'all_img': all_img,
#                   'all_fc_act':all_fc_act,
#                   'all_rnn_act':all_rnn_act,
#                   'all_v_act':all_v_act,
#                   'all_actions':all_actions,
#                   'all_health':all_health,
#                   'conv_acts': conv_acts}
#
#    np.save(f'{os.getcwd()}/train_dir/{cfg.experiment}/analyze_out.npy', analyze_out, allow_pickle=True)
#
#def load_sim_out(cfg):
#    sim_out = np.load(f'{os.getcwd()}/train_dir/{cfg.experiment}/analyze_out.npy', allow_pickle=True).tolist()
#    return sim_out
#
#def get_acts_dataset(cfg, actor_critic):
#
#    bck_np = np.load(os.getcwd() + '/misc/data/doom_pad.npy') # saved 'doom-looking' background
#
#    if cfg.analyze_ds_name == 'CIFAR':
#        trainset = datasets.CIFAR10(root='./data', train=True, download=True, transform=None)
#        testset = datasets.CIFAR10(root='./data', train=False, download=True, transform=None)
#        rewards_dict = {0:2, 1:1, 2:4, 3:5, 4:7, 5:6, 6:8, 7:9, 8:3, 9:0} # defined by the class-reward assignments in the .wad file
#
#    elif cfg.analyze_ds_name == 'MNIST':
#        trainset = datasets.MNIST(root='./data', train=True, download=True, transform=None)
#        testset = datasets.MNIST(root='./data', train=False, download=True, transform=None)
#        rewards_dict = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:9} # defined by the class-reward assignments in the .wad file (matching digits in case of mnist)
#
#    n_stim = len(trainset)
#    all_fc_act = np.zeros((cfg.hidden_size, n_stim))
#    all_img = np.zeros((cfg.res_h, cfg.res_w, 3, n_stim))
#    all_lab = np.zeros(n_stim)
#
#    with torch.no_grad():
#        for i in range(n_stim):
#                obs = pad_dataset(cfg, i, trainset, bck_np, offset)
#                fc_act_torch = actor_critic.encoder.base_encoder.forward(obs) # activation of output fc layer
#
#                all_fc_act[:,i] = fc_act_torch.cpu().detach().numpy()
#                all_img[:,:,:,i] = obs_to_img(obs.cpu()).astype(np.uint8)
#                all_lab[i] = rewards_dict[trainset[i][1]] # getting label for sample and converting based on health assignment
#
#                # progress
#                if i % int(n_stim/20) == 0:
#                    print(f'Forward pass through {i}/{n_stim} dataset entries')
#
#    analyze_ds_out = {'all_img': all_img,
#                    'all_fc_act':all_fc_act,
#                   'all_lab':all_lab}
#
#    return analyze_ds_out
#
#def obs_to_img(obs):
#    img = np.array(obs[0]).astype(np.uint8)
#    img = np.transpose(img, (1,2,0))
#    return img
#
#def pad_dataset(cfg, i, trainset, bck_np): # i:index in dataset, bck_np: numpy array of background (grass/sky), offset:determines position within background
#    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
#    in_im = trainset[i][0]
#    if cfg.analyze_ds_name == 'CIFAR':
#        im_hw = 32
#        in_np = np.array(np.transpose(in_im, (2,0,1)))
#        offset = (28,50)
#    elif cfg.analyze_ds_name == 'MNIST':
#        im_hw = 28
#        in_np = np.array(in_im)*256 # originally they're normalized between 0 and 1
#        offset = (40,50)
#    out_np = bck_np # background
#    out_np[:,offset[0]:offset[0]+im_hw, offset[1]:offset[1]+im_hw] = in_np # replacing pixels in the middle
#    out_np_t = np.transpose(out_np, (1, 2, 0)) # reformatting for PIL conversion
#    out_im = im.fromarray(out_np_t)
#    out_torch = torch.from_numpy(out_np[None,:,:,:]).float().to(device)
#
#    return out_torch
#
#def unroll_conv_acts(conv_acts, lay=1):
#
#    acts = np.array(conv_acts[lay])
#
#    n_px = acts.shape[1] * acts.shape[2] # new dimension after flattening pixels
#    n_ts = acts.shape[0]
#    n_ch = acts.shape[3]
#
#    unroll_acts = acts.reshape(n_ts, n_px, n_ch)
#    return unroll_acts
#
### from here on are post-Cosyne functions/classes (action-analysis and explainability)
#
#
#
#class PaddedMnistDataset(Dataset):
#    def __init__(self, padded_ds, labels_ds):
#        self.all_img = padded_ds
#        self.all_lab = labels_ds
#
#    def __len__(self):
#        return len(self.all_lab)
#
#    def __getitem__(self, idx):
#        image = self.all_img[:,:,:,idx]
#        label = self.all_lab[idx]
#
#        return image, label
#
#class PytorchActorCritic(nn.Module):
#    def __init__(self, cfg, actor_critic):
#        super().__init__()
#
#        self.cfg = cfg
#
#        self.conv_head_out_size = actor_critic.encoder.conv_head_out_size
#
#        self.conv_head = actor_critic.encoder.base_encoder.conv_head
#        self.fc1 = actor_critic.encoder.base_encoder.fc1 # here we will need to flatten the features before going forward
#        self.nl_fc = actor_critic.encoder.base_encoder.nl_fc
#
#        self.critic = actor_critic.critic_linear
#        self.actor = actor_critic.action_parameterization.distribution_linear # here we will need two separate networks, for value and actions
#
#
#    def forward(self, x):
#        # conv layer 1
#        normalize_obs_torch(x, self.cfg) # this is ommitted by the encoder, but implemented in the forward_head() method of the AC object
#
#        x = self.conv_head(x)
#        x = x.contiguous().view(-1, self.conv_head_out_size)
#
#        x = self.fc1(x)
#        x = self.nl_fc(x)
#
#        # here there would normally be the forward pass on the core, but this is not necessary for a purely feed-forward network
#
#        if self.cfg.a_or_c == 'a':
#            x = self.actor(x)
#        elif self.cfg.a_or_c == 'c':
#            x = self.critic(x)
#
#        return x
#
#def normalize_obs_torch(obs_torch, cfg):
#    with torch.no_grad():
#        mean = cfg.obs_subtract_mean
#        scale = cfg.obs_scale
#
#        if obs_torch.dtype != torch.float:
#            obs_torch = obs_torch.float()
#
#        if abs(mean) > EPS:
#            obs_torch.sub_(mean)
#
#        if abs(scale - 1.0) > EPS:
#            obs_torch.mul_(1.0 / scale)
#
#class LindseyClassificationEncoder(LindseyEncoderBase): # this is almost the same class definition as LindseyEncoder with only two small additions (*)
#
#    def __init__(self, cfg, obs_space,timing):
#
#        super().__init__(cfg,obs_space,timing)
#        self.base_encoder = LindseyEncoderBase(cfg,obs_space,timing)
#        self.fc_out = nn.Linear(self.encoder_out_size,10) # (*) addition for classification
#        print(self)
#
#    def forward(self, obs_dict):
#        # we always work with dictionary observations. Primary observation is available with the key 'obs'
#        main_obs = obs_dict#['obs']
#
#        # forward pass through configurable fully connected blocks immediately after the encoder
#        x = self.base_encoder(main_obs)
#        x = self.fc_out(x) # (*) addition for classification
#        return x
#
## this one is re-defined here to overwrite the original version from the library
#def pad_dataset_attribution(cfg, i, trainset, bck_np, downs_fact=1, mnist_offset=(50,46)): # i:index in dataset, bck_np: numpy array of background (grass/sky), offset:determines position within background
#    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
#    in_im = trainset[i][0]
#    in_im = in_im[:,::downs_fact, ::downs_fact] # downsampling if downs_fact > 1
#    if cfg.analyze_ds_name == 'CIFAR':
#        im_hw = int(32/downs_fact)
#        in_np = np.array(np.transpose(in_im, (2,0,1)))
#        offset = (28,50)
#    elif cfg.analyze_ds_name == 'MNIST':
#        im_hw = int(28/downs_fact)
#        in_np = np.array(in_im)*256 # originally they're normalized between 0 and 1
#        offset = mnist_offset
#    out_np = bck_np # background
#    out_np[:,offset[0]:offset[0]+im_hw, offset[1]:offset[1]+im_hw] = in_np # replacing pixels in the middle
#    out_np_t = np.transpose(out_np, (1, 2, 0)) # reformatting for PIL conversion
#    out_im = im.fromarray(out_np_t)
#    out_torch = torch.from_numpy(out_np[None,:,:,:]).float().to(device)
#
#    return out_torch
#
#def get_padded_mnist(cfg, downs_fact=1, mnist_offset=(50,46)):
#    mnist_train = datasets.MNIST(root="./datasets", train=True, transform=transforms.ToTensor(), download=True)
#    mnist_test = datasets.MNIST(root="./datasets", train=False, transform=transforms.ToTensor(), download=True)
#
#    bck_np = np.load(os.getcwd() + '/misc/data/doom_pad.npy') # saved 'doom-looking' background
#
#    padded_ds = torch.zeros((3, cfg.res_h, cfg.res_w, len(mnist_train)))
#    labels_ds = torch.zeros(len(mnist_train))
#
#    for i in range(len(mnist_train)):
#        out = pad_dataset_attribution(cfg, i, mnist_train, bck_np, downs_fact=downs_fact, mnist_offset=mnist_offset).squeeze() #
#        padded_ds[:,:,:,i] = out
#        labels_ds[i] = mnist_train[i][1]
#
#    padded_ds = PaddedMnistDataset(padded_ds, labels_ds)
#
#    return padded_ds
#
#def get_all_action(cfg, env, padded_ds, actor_critic, get_raw_logits=False):
#    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
#    n_stim = len(padded_ds)
#    all_label = np.zeros(n_stim)
#    all_action = np.zeros((actor_critic.actor.out_features, n_stim))
#
#    for im_idx in range(n_stim):
#
#        # preparing image
#        in_image = padded_ds[im_idx][0]
#        in_label = padded_ds[im_idx][1].item()
#        all_label[im_idx] = in_label
#
#        # forward pass on actor
#        action = actor_critic(in_image.to(device))
#        action_dist = get_action_distribution(env.action_space, raw_logits=action)
#        action_dist_p0 = action_dist.distributions[0].probs
#        action_dist_p1 = action_dist.distributions[1].probs
#
#        action_dist_all = np.hstack((action_dist_p0.cpu().detach().numpy()[0], action_dist_p1.cpu().detach().numpy()[0]))
#        all_action[:,im_idx] = action_dist_all
#
#        if get_raw_logits: # overwrite the 0-1 bounded (post softmax) action distribution by the unbounded raw logits before softmax
#            all_action[:,im_idx] = action.cpu().detach().numpy()
#
#    return all_action, all_label
