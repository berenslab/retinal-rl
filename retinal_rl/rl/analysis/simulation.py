### Util for preparing simulations and data for analysis

from typing import Tuple
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import torch

torch.backends.cudnn.enabled=False

from tqdm.auto import tqdm

from sample_factory.algo.learning.learner import Learner
from sample_factory.algo.sampling.batched_sampling import preprocess_actions
from sample_factory.algo.utils.action_distributions import argmax_actions
from sample_factory.algo.utils.make_env import make_env_func_batched,BatchedVecEnv
from sample_factory.algo.utils.rl_utils import prepare_and_normalize_obs
from sample_factory.algo.utils.tensor_utils import unsqueeze_tensor
from sample_factory.cfg.arguments import load_from_checkpoint
from sample_factory.model.actor_critic import create_actor_critic,ActorCritic
from sample_factory.model.model_utils import get_rnn_size
from sample_factory.utils.attr_dict import AttrDict
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.env_info import extract_env_info
from sample_factory.utils.utils import log

from retinal_rl.util import obs_dict_to_tuple,obs_to_img,from_float_to_rgb

def get_checkpoint(cfg: Config):
    """
    Load the model from checkpoint, initialize the environment, and return both.
    """
    #verbose = False

    cfg = load_from_checkpoint(cfg)

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")

    policy_id = cfg.policy_index
    name_prefix = dict(latest="checkpoint", best="best")[cfg.load_checkpoint_kind]
    checkpoints = Learner.get_checkpoints(Learner.checkpoint_dir(cfg, policy_id), f"{name_prefix}_*")
    checkpoint_dict = Learner.load_checkpoint(checkpoints, device)

    return checkpoint_dict,cfg

def get_brain_env(cfg: Config, checkpoint_dict) -> Tuple[ActorCritic,BatchedVecEnv,AttrDict,int]:
    """
    Load the model from checkpoint, initialize the environment, and return both.
    """
    #verbose = False

    cfg.env_frameskip = cfg.eval_env_frameskip

    cfg.num_envs = 1

    # In general we only focus on saving to files
    render_mode = "rgb_array"

    log.debug("RETINAL RL: Making environment...")
    env = make_env_func_batched(
        cfg, env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode=render_mode
    )

    log.debug("RETINAL RL: Finished making environment, loading actor-critic model...")
    brain = create_actor_critic(cfg, env.observation_space, env.action_space)
    # log.debug("RETINAL RL: ...evaluating actor-critic model...")
    brain.eval()

    # log.debug("RETINAL RL: Actor-critic initialized...")

    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")
    brain.model_to_device(device)

    # log.debug("RETINAL RL: ...copied to device...")

    brain.load_state_dict(checkpoint_dict["model"])
    nstps = checkpoint_dict["env_steps"]

    # log.debug("RETINAL RL: ...and loaded from checkpoint.")

    return brain,env,cfg,nstps

def generate_simulation(cfg: Config, brain : ActorCritic, env : BatchedVecEnv, sim_recs, prgrs : bool, video : bool):
    """
    Save an example simulation.
    """
    num_frames = 0
    t_max = int(cfg.max_num_frames)

    if sim_recs is None:

        sim_recs = {}
        # Initializing stream arrays
        sim_recs['ltnts'] = np.zeros((cfg.rnn_size, t_max))
        sim_recs['plcys'] = np.zeros((2,3, t_max))
        sim_recs['uhlths'] = np.zeros(t_max)
        sim_recs['nnrshms'] = np.zeros(t_max)
        sim_recs['npsns'] = np.zeros(t_max)
        sim_recs['hlths'] = np.zeros(t_max)
        sim_recs['rwds'] = np.zeros(t_max)
        sim_recs['vals'] = np.zeros(t_max)
        sim_recs['crwds'] = np.zeros(t_max)
        sim_recs['dns'] = np.zeros(t_max)

        if video:

            sim_recs['imgs'] = np.zeros((cfg.res_h, cfg.res_w, 3, t_max)).astype(np.uint8)
            sim_recs['nimgs'] = np.zeros((cfg.res_h, cfg.res_w, 3, t_max))
            sim_recs['attrs'] = np.zeros((cfg.res_h, cfg.res_w, 3, t_max))

    else:
        # Set num_frames to the number of frames already saved
        num_frames = sim_recs['rwds'].shape[0]
        # Extend the arrays to the new size
        sim_recs['ltnts'] = np.concatenate((sim_recs['ltnts'],np.zeros((cfg.rnn_size, t_max))),axis=1)
        sim_recs['plcys'] = np.concatenate((sim_recs['plcys'],np.zeros((2,3, t_max))),axis=2)
        sim_recs['uhlths'] = np.concatenate((sim_recs['uhlths'],np.zeros(t_max)))
        sim_recs['nnrshms'] = np.concatenate((sim_recs['nnrshms'],np.zeros(t_max)))
        sim_recs['npsns'] = np.concatenate((sim_recs['npsns'],np.zeros(t_max)))
        sim_recs['hlths'] = np.concatenate((sim_recs['hlths'],np.zeros(t_max)))
        sim_recs['rwds'] = np.concatenate((sim_recs['rwds'],np.zeros(t_max)))
        sim_recs['vals'] = np.concatenate((sim_recs['vals'],np.zeros(t_max)))
        sim_recs['crwds'] = np.concatenate((sim_recs['crwds'],np.zeros(t_max)))
        sim_recs['dns'] = np.concatenate((sim_recs['dns'],np.zeros(t_max)))

        if video:

            sim_recs['imgs'] = np.cat((sim_recs['imgs'],np.zeros((cfg.res_h, cfg.res_w, 3, t_max-num_frames)).astype(np.uint8)),dim=3)
            sim_recs['nimgs'] = np.cat((sim_recs['nimgs'],np.zeros((cfg.res_h, cfg.res_w, 3, t_max-num_frames))),dim=3)
            sim_recs['attrs'] = np.cat((sim_recs['attrs'],np.zeros((cfg.res_h, cfg.res_w, 3, t_max-num_frames))),dim=3)

    t_max = int(cfg.max_num_frames) + num_frames

    # Initializing some local variables
    env_info = extract_env_info(env, cfg)
    action_repeat: int = cfg.env_frameskip // cfg.eval_env_frameskip
    device = torch.device("cpu" if cfg.device == "cpu" else "cuda")


    # Initializing simulation state
    obs_dict,_ = env.reset()
    nobs_dict = prepare_and_normalize_obs(brain, obs_dict)
    nobs,msms = obs_dict_to_tuple(nobs_dict)
    rnn_states = torch.zeros([env.num_agents, get_rnn_size(cfg)], dtype=torch.float32, device=device)
    is_dn=0
    rwd=0
    crwd=0

    nobs1 = torch.unsqueeze(nobs,0)
    msms1 = torch.tensor([])

    if msms is not None:
        msms1 = torch.unsqueeze(msms,0)

    # onnx_inpts = brain.prune_inputs(nobs1,msms1,rnn_states)

    # Simulation loop with tqdm
    for num_frames in tqdm(range(num_frames,t_max), disable=not prgrs):

        # Evaluate policy
        policy_outputs = brain(nobs_dict, rnn_states)
        rnn_states = policy_outputs["new_rnn_states"]
        actions = policy_outputs["actions"]
        action_distribution = brain.action_distribution()
        acts_dstrbs = action_distribution.distributions
        policy = torch.stack([dstrb.probs[0] for dstrb in acts_dstrbs])
        value = policy_outputs["values"]

        ltnt = policy_outputs["latent_states"]

        # can pass --eval_deterministic=True to CLI in order to argmax the probabilistic actions
        if cfg.eval_deterministic:
            action_distribution = brain.action_distribution()
            actions = argmax_actions(action_distribution)

        # actions shape should be [num_agents, num_actions] even if it's [1, 1]
        if actions.ndim == 1:
            actions = unsqueeze_tensor(actions, dim=-1)
        actions = preprocess_actions(env_info, actions)

        # Repeating actions during evaluation because we run the simulation at higher FPS
        for _ in range(action_repeat):

            obs,msms = obs_dict_to_tuple(obs_dict)
            nobs,_ = obs_dict_to_tuple(nobs_dict)

            info = env.unwrapped.get_info()
            health = info.get('HEALTH') # environment info (health etc.)
            unbound_health = info.get('USER17') # environment info (health etc.)
            num_nourishments = info.get('USER18') # environment info (health etc.)
            num_poisons = info.get('USER19') # environment info (health etc.)

            if is_dn:
                crwd=0
            else:
                crwd+=rwd

            sim_recs['ltnts'][:,num_frames] = ltnt.cpu().detach().numpy()
            sim_recs['plcys'][:,:,num_frames] = policy.cpu().detach().numpy()

            sim_recs['hlths'][num_frames] = health
            sim_recs['uhlths'][num_frames] = unbound_health
            sim_recs['nnrshms'][num_frames] = num_nourishments
            sim_recs['npsns'][num_frames] = num_poisons
            sim_recs['dns'][num_frames] = is_dn
            sim_recs['vals'][num_frames] = value
            sim_recs['rwds'][num_frames] = rwd
            sim_recs['crwds'][num_frames] = crwd

            if video:

                if msms is not None:
                    msms1 = torch.unsqueeze(msms,0)

                nobs1 = torch.unsqueeze(nobs,0)
                nobsatt = brain.attribute(nobs1, msms1, rnn_states)
                attr = torch.squeeze(nobsatt,0)

                img = obs_to_img(obs)
                nimg = obs_to_img(nobs)
                attrimg = obs_to_img(attr)

                sim_recs['imgs'][:,:,:,num_frames] = img
                sim_recs['nimgs'][:,:,:,num_frames] = nimg
                sim_recs['attrs'][:,:,:,num_frames] = attrimg

            obs_dict,rwd,terminated,truncated,_ = env.step(actions)
            is_dn = truncated | terminated
            nobs_dict = prepare_and_normalize_obs(brain, obs_dict)

            # Report multivariate mean and std of plcys from 0 to current frame using tqdm
            # if prgrs and num_frames % 100 == 0:
            #     heading_means = np.mean(plcys[0,:,:num_frames+1],axis=1)
            #     heading_sds = np.std(plcys[0,:,:num_frames+1],axis=1)
            #     velocity_means = np.mean(plcys[1,:,:num_frames+1],axis=1)
            #     velocity_sds = np.std(plcys[1,:,:num_frames+1],axis=1)
            #   # Update the statistics on the second line
            #     stats1.set_description(f"Heading stats: means {heading_means}; sds: {heading_sds}")
            #     stats1.refresh()  # Refresh to show the updated statistics
            #     stats2.set_description(f"Velocity stats: means {velocity_means}; sds: {velocity_sds}")
            #     stats2.refresh()  # Refresh to show the updated statistics
            #

    if video:
        sim_recs['attrs'] = abs(sim_recs['attrs'])
        sim_recs['attrs'] = from_float_to_rgb(sim_recs['attrs'])
        sim_recs['nimgs'] = from_float_to_rgb(sim_recs['nimgs'])

    # Shuffle x and y in the same way
    # idx = shuffle(np.arange(t_max))
    # x = hlths[idx]
    # # transpose ltnts so that each row is a latent state
    # y = np.transpose(ltnts)
    # y = y[idx]
    #
    # # Split data into training and test sets
    # x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
    #
    # # Train a linear predictor
    # reg = LinearRegression().fit(y_train, x_train)
    #
    # # Predict on the test set
    # x_pred = reg.predict(y_test)
    #
    # # Calculate the root mse
    # rmse = mean_squared_error(x_test, x_pred, squared=False)
    #
    # print(f"\n\n\nHealth Mean Squared Error: {rmse}")
    # print(f"Coefficients: {reg.coef_}")
    #
    return sim_recs



