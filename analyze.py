import sys

import time
import torch
import numpy as np
import matplotlib.pyplot as plt
#from torchinfo import summary
#from captum.attr import NeuronGradient

from sample_factory.algorithms.appo.model_utils import get_hidden_size
from sample_factory.algorithms.appo.learner import LearnerWorker
from sample_factory.algorithms.appo.model import create_actor_critic
from sample_factory.algorithms.utils.arguments import parse_args, load_from_checkpoint
from sample_factory.algorithms.utils.multi_agent_wrapper import MultiAgentWrapper, is_multiagent_env
from sample_factory.envs.create_env import create_env
from sample_factory.utils.utils import log, AttrDict
from sample_factory.algorithms.appo.actor_worker import transform_dict_observations

from retina_rl.retina_rl import register_custom_components, custom_parse_args

def io_size(dev,env,enc):

    obs = env.reset()
    obs_torch = AttrDict(transform_dict_observations(obs))
    for key, x in obs_torch.items():
        obs_torch[key] = torch.from_numpy(x).to(dev).float()
    isz = list(obs_torch['obs'].size())[1:]
    outmtx = enc.nl2(enc.conv2(enc.nl1(enc.conv1(obs_torch['obs']))))
    osz = list(outmtx.size())[1:]

    return isz,osz

def spike_triggered_average(dev,enc,flt,rds,isz):

    with torch.no_grad():

        btchsz = [50000] + isz
        cnty = btchsz[2]//2
        cntx = btchsz[3]//2
        mny = cnty - rds
        mxy = cnty + rds+1
        mnx = cntx - rds
        mxx = cntx + rds+1
        obsns = torch.randn(size=btchsz,device=dev)
        outmtx = enc.nl2(enc.conv2(enc.nl1(enc.conv1(obsns))))
        outsz = outmtx.size()
        outs = outmtx[:,flt,outsz[2]//2,outsz[3]//2].cpu()
        obsns1 = obsns[:,:,mny:mxy,mnx:mxx].cpu()
        avg = np.average(obsns1,axis=0,weights=outs)

    return avg


def analyze(cfg):
    # Calling up the encoder
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

    #device = torch.device('cpu')
    device = torch.device('cpu' if cfg.device == 'cpu' else 'cuda')
    actor_critic.model_to_device(device)

    policy_id = cfg.policy_index
    checkpoints = LearnerWorker.get_checkpoints(LearnerWorker.checkpoint_dir(cfg, policy_id))
    checkpoint_dict = LearnerWorker.load_checkpoint(checkpoints, device)
    actor_critic.load_state_dict(checkpoint_dict['model'])
    enc = actor_critic.encoder.cnn_encoder

    # Printing encoder stats
    print(enc)

    # Analysing encoder
    isz,osz = io_size(device,env,enc)
    print("Input Size: ", isz)
    print("Output Size: ", osz)
    nchns = isz[0]
    flts = osz[0]
    rds = 3
    rwsmlt = 2
    fltsdv = flts//rwsmlt

    fig, axs = plt.subplots(nchns*rwsmlt,fltsdv)

    for i in range(fltsdv):

        for j in range(rwsmlt):

            flt = i + j*fltsdv
            avg = spike_triggered_average(device,enc,flt,rds,isz)

            for k in range(nchns):

                # Plotting statistics
                rw = k + j*nchns
                ax = axs[rw,i]
                vmx = abs(avg[k,:,:]).max()
                pnl = ax.imshow(avg[k,:,:],vmin=-vmx,vmax=vmx)
                fig.colorbar(pnl, ax=ax)

                if k == 0:
                    ax.set_title("Filter: " + str(flt), { 'weight' : 'bold' } )

    plt.show()

def main():
    """Script entry point."""
    register_custom_components()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())
