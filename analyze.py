import sys

from retinal_rl.system.environment import register_retinal_environment
from retinal_rl.system.encoders import register_encoders
from retinal_rl.system.parameters import custom_parse_args

from retinal_rl.analysis.util import get_env_ac, simulate, load_sim_out, get_acts_dataset, unroll_conv_acts
from retinal_rl.analysis.statistics import get_class_accuracy
from retinal_rl.analysis.plot import save_simulation_gif, plot_all_rf, plot_acts_tsne_stim, plot_dimred_ds_acts, save_activations_gif

def analyze(cfg):
    env, actor_critic = get_env_ac(cfg)
    simulate(cfg, env, actor_critic) # this saves

    # load simulated data
    sim_out = load_sim_out(cfg)

    # visualise
    plot_all_rf(cfg, actor_critic, env) # receptive fields
    save_simulation_gif(cfg, sim_out['all_img'])

    plot_acts_tsne_stim(cfg, sim_out['all_fc_act'], sim_out['all_health'], title='FC')
    plot_acts_tsne_stim(cfg, sim_out['all_rnn_act'], sim_out['all_health'], title='RNN')

    lay = 1 # for now we only care about bottleneck layer
    # save_activations_gif(cfg, sim_out['all_img'], sim_out['conv_acts'], lay, vscale=10)
    unroll_acts = unroll_conv_acts(sim_out['conv_acts'], lay=lay)
    for ch in range (unroll_acts.shape[2]):
        plot_acts_tsne_stim(cfg, unroll_acts[:,:,ch].T, sim_out['all_health'], title=f'l{lay}_ch{ch}')

    #plot_dimred_sim_acts(cfg, sim_out['all_fc_act'], title='FC') # these are embeddings of the activation time-series (not as informative/would need longer simulations)
    #plot_dimred_sim_acts(cfg, sim_out['all_rnn_act'], title='RNN')

    if cfg.analyze_acts == 'dataset':
        ds_out = get_acts_dataset(cfg, actor_critic)
        plot_dimred_ds_acts(cfg, ds_out['all_fc_act'], ds_out['all_lab'])

        for mode in ['multi', 'bin']:
            for permute in [False, True]:
                out_str = get_class_accuracy(cfg, ds_out, mode=mode, permute=permute)
                print(out_str) # ADD SAVING THIS STRING


def main():
    """Script entry point."""
    register_retinal_environment()
    register_encoders()
    cfg = custom_parse_args(evaluation=True)
    analyze(cfg)


if __name__ == '__main__':
    sys.exit(main())
