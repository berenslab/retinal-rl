import torch
import sys
import os
import wandb
import multiprocessing
import matplotlib.pyplot as plt

multiprocessing.set_start_method("spawn", force=True) # Important.  TODO: Readup on this


from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.misc import ExperimentStatus

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.utils import log, debug_log_every_n

from retinal_rl.rl.system.brain import register_brain
from retinal_rl.rl.system.environment import register_retinal_env

from retinal_rl.util import (
    analysis_root,
    plot_path,
    write_analysis_count,
    read_analysis_count,
)

from retinal_rl.rl.analysis.simulation import (
    get_brain_env,
    generate_simulation,
    get_checkpoint,
)
from retinal_rl.rl.analysis.statistics import (
    gaussian_noise_stas,
    gradient_receptive_fields,
)
from retinal_rl.util import (
    save_data,
    load_data,
    analysis_path,
    plot_path,
    data_path,
)

from retinal_rl.rl.analysis.plot import simulation_plot, receptive_field_plots, PNGWriter

### Analysis ###


def analyze(cfg, progress_bar=True):

    register_retinal_env(cfg.env, cache_dir=os.path.join(os.getcwd(), "cache"), input_satiety=cfg.input_satiety)
    register_brain()

    log.debug(
        "Running analysis: simulate = %s, plot = %s, animate = %s",
        cfg.simulate,
        cfg.plot,
        cfg.animate,
    )

    # Register retinal environments and models.
    checkpoint_dict, cfg = get_checkpoint(cfg)

    if checkpoint_dict is None:
        log.debug("RETINAL RL: No checkpoint found, aborting analysis.")
        sys.exit(1)

    log.debug("RETINAL RL: Checkpoint loaded, preparing environment.")

    brain, env, cfg, envstps = get_brain_env(cfg, checkpoint_dict)

    if cfg.analysis_name is None:
        ana_name = "env_steps-" + str(envstps)
    else:
        ana_name = cfg.analysis_name

    if not os.path.exists(analysis_path(cfg, ana_name)):
        os.makedirs(data_path(cfg, ana_name))

    rf_algs = ["grads"]
    log.debug("RETINAL RL: Model and environment loaded, preparing simulation.")

    """ Final gluing together of all analyses of interest. """
    if cfg.simulate:

        log.debug("RETINAL RL: Running analysis simulations.")

        sim_recs = None

        if cfg.append_sim:
            # Check if sim_recs exists, if not, create it
            if os.path.exists(data_path(cfg, ana_name, "sim_recs.npy")):
                sim_recs = load_data(cfg, ana_name, "sim_recs")

        sim_recs = generate_simulation(
            cfg, brain, env, sim_recs, prgrs=progress_bar, video=cfg.viewport_video
        )

        # save_onnx(cfg,ana_name,brain,inpts)

        save_data(cfg, ana_name, sim_recs, "sim_recs")

    if cfg.receptive_fields:

        log.debug("RETINAL RL: Analyzing receptive fields.")

        for alg in rf_algs:

            if not os.path.exists(os.path.join(plot_path(cfg, ana_name), alg + "_rfs")):
                os.makedirs(os.path.join(plot_path(cfg, ana_name), alg + "_rfs"))

            if alg == "stas":
                stas = gaussian_noise_stas(
                    cfg,
                    env,
                    brain,
                    nbtch=200,
                    nreps=cfg.sta_repeats,
                    prgrs=progress_bar,
                )
                save_data(cfg, ana_name, stas, "stas")

            if alg == "grads":
                grads = gradient_receptive_fields(cfg, env, brain, prgrs=progress_bar)
                save_data(cfg, ana_name, grads, "grads")

    if cfg.plot:

        log.debug("RETINAL RL: Plotting analysis simulations.")

        # Load data
        sim_recs = load_data(cfg, ana_name, "sim_recs")

        # Single frame of the animation
        # fig = plot_acts_tsne_stim(sim_recs)
        # pth=plot_path(cfg,ana_name,"latent-activations.png")

        # fig.savefig(pth, bbox_inches="tight")
        # plt.close()

        # Single frame of the animation
        if cfg.viewport_video:
            fig = simulation_plot(
                sim_recs, frame_step=cfg.frame_step, prgrs=progress_bar
            )
            pth = plot_path(cfg, ana_name, "simulation-frame.png")
            fig.savefig(pth, bbox_inches="tight")
            plt.close()

        for alg in rf_algs:

            rfs = load_data(cfg, ana_name, alg)

            for ky in rfs.keys():

                lyr = rfs[ky]
                fig = receptive_field_plots(lyr)
                fig.savefig(
                    plot_path(cfg, ana_name, alg + "_rfs/" + alg + "-" + ky + ".png"),
                    bbox_inches="tight",
                )
                plt.close()

    if cfg.animate:

        log.debug("RETINAL RL: Animating analysis simulations.")

        # Animation
        sim_recs = load_data(cfg, ana_name, "sim_recs")
        anim = simulation_plot(sim_recs, animate=True, fps=cfg.fps, prgrs=progress_bar)
        if cfg.save_frames:
            # Save animation frames as individual pngs in a subfolder
            pth = plot_path(cfg, ana_name, "simulation-frames")
            if not os.path.exists(pth):
                os.makedirs(pth)
            writer = PNGWriter()
            anim.save(pth, writer=writer)
        else:
            pth = plot_path(cfg, ana_name, "simulation-animation.mp4")
            anim.sa.rlve(pth, extra_args=["-vcodec", "libx264"])

    env.close()

    return envstps


### Runner ###


class RetinalAlgoObserver(AlgoObserver):
    """
    AlgoObserver that runs analysis at specified times.
    """

    def __init__(self, cfg):
        self.cfg = cfg
        self.freq = cfg.analysis_freq
        self.current_process = None
        self.queue = multiprocessing.Queue()

        # get analysis count
        if not os.path.exists(analysis_root(cfg)):
            os.makedirs(analysis_root(cfg))

        acount = read_analysis_count(cfg)

        self.steps_complete = acount

    def analyze(self, queue):
        """Run analysis in a separate process."""

        envstps = analyze(self.cfg, progress_bar=False)
        queue.put(envstps, block=False)

    def on_training_step(self, runner: Runner, _) -> None:
        """Called after each training step."""

        if self.current_process is None:

            total_env_steps = sum(runner.env_steps.values())
            current_step = total_env_steps // self.freq

            msg = (
                "RETINAL RL: No analysis running. current_step = %d, steps_complete = %d"
                % (current_step, self.steps_complete)
            )
            debug_log_every_n(100, msg)

            if current_step >= self.steps_complete:
                # run analysis in a separate process
                log.debug(
                    "RETINAL RL: current_step >= self.steps_complete, launching analysis process..."
                )
                self.current_process = multiprocessing.Process(
                    target=self.analyze, args=(self.queue,)
                )
                self.current_process.start()

        else:
            if not self.current_process.is_alive():

                if self.current_process.exitcode == 0:

                    log.debug(
                        "RETINAL RL: Analysis process finished successfully. Retrieving envstps..."
                    )
                    envstps = self.queue.get()
                    ana_name = "env_steps-" + str(envstps)

                    if self.cfg.with_wandb:
                        log.debug("RETINAL RL: Uploading plots to wandb...")

                        pltpth = plot_path(self.cfg, ana_name)
                        # Recursively list all files in pltpth
                        for path, _, files in os.walk(pltpth):
                            # upload all pngs to wandb
                            for f in files:
                                if f.endswith(".png"):
                                    log.debug("RETINAL RL: Uploading %s", f)
                                    wandb.log({f: wandb.Image(os.path.join(path, f))})
                            # Upload video to wandb
                            for f in files:
                                if f.endswith(".mp4"):
                                    log.debug("RETINAL RL: Uploading %s", f)
                                    wandb.log({f: wandb.Video(os.path.join(path, f))})

                        # for f in os.listdir(pltpth):
                        #    if f.endswith(".png"):
                        #        log.debug("RETINAL RL: Uploading %s",f)
                        #        wandb.log({f: wandb.Image(os.path.join(pltpth,f))})
                        ## load all mp4 files in the plot directory and upload them to wandb
                        # for f in os.listdir(pltpth):
                        #    if f.endswith(".mp4"):
                        #        log.debug("RETINAL RL: Uploading %s",f)
                        #        wandb.log({f: wandb.Video(os.path.join(pltpth,f))})

                    self.steps_complete += 1
                    write_analysis_count(self.cfg, self.steps_complete)

                self.current_process.join()
                self.current_process = None


def run_rl(cfg: Config):
    """Run RL training."""

    cfg, runner = make_runner(cfg)
    if cfg.online_analysis:
        runner.register_observer(RetinalAlgoObserver(cfg))

    # here we can register additional message or summary handlers
    # see sf_examples/dmlab/train_dmlab.py for example

    status = runner.init()
    if status == ExperimentStatus.SUCCESS:
        status = runner.run()

    return status
