import sys
import os
import wandb
import multiprocessing
multiprocessing.set_start_method("spawn",force=True)
import torchscan as ts


from sample_factory.cfg.arguments import parse_sf_args, parse_full_cfg
from sample_factory.train import make_runner
from sample_factory.utils.typing import Config
from sample_factory.algo.utils.misc import ExperimentStatus
from sample_factory.utils.attr_dict import AttrDict

from sample_factory.algo.runners.runner import AlgoObserver, Runner
from sample_factory.utils.utils import log,debug_log_every_n
from sample_factory.algo.utils.make_env import make_env_func_batched

from retinal_rl.system.encoders import register_retinal_model,make_network
from retinal_rl.system.environment import register_retinal_env
from retinal_rl.system.arguments import retinal_override_defaults,add_retinal_env_args,add_retinal_env_eval_args

from retinal_rl.util import get_analysis_times,analysis_root,plot_path

from analyze import analyze


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

        # get analysis times
        if not os.path.exists(analysis_root(cfg)):
            os.makedirs(analysis_root(cfg))

        self.analysis_times = get_analysis_times(cfg)

        self.last_analysis = max(self.analysis_times,default=-1)
        self.steps_complete = 1 + self.last_analysis // self.freq

    def analyze(self,queue):
        """Run analysis in a separate process."""

        envstps = analyze(self.cfg,progress_bar=False)
        queue.put(envstps,block=False)

    def on_training_step(self, runner: Runner, _) -> None:
        """Called after each training step."""

        if self.current_process is None:

            total_env_steps = sum(runner.env_steps.values())
            current_step = total_env_steps // self.freq

            msg = "RETINAL RL: No analysis running. current_step = %d, steps_complete = %d" % (current_step,self.steps_complete)
            debug_log_every_n(100,msg)

            if current_step >= self.steps_complete:
                # run analysis in a separate process
                log.debug("RETINAL RL: current_step >= self.steps_complete, launching analysis process...")
                self.current_process = multiprocessing.Process(target=self.analyze,args=(self.queue,))
                self.current_process.start()

        else:
            if not self.current_process.is_alive():

                if self.current_process.exitcode == 0:

                    log.debug("RETINAL RL: Analysis process finished successfully. Retrieving envstps...")
                    envstps = self.queue.get()
                    ana_name = "env_steps-" + str(envstps)

                    if self.cfg.with_wandb:
                        log.debug("RETINAL RL: Uploading plots to wandb...")

                        pltpth = plot_path(self.cfg,ana_name)
                        # Recursively list all files in pltpth
                        for path, _, files in os.walk(pltpth):
                            # upload all pngs to wandb
                            for f in files:
                                if f.endswith(".png"):
                                    log.debug("RETINAL RL: Uploading %s",f)
                                    wandb.log({f: wandb.Image(os.path.join(path,f))})
                            # Upload video to wandb
                            for f in files:
                                if f.endswith(".mp4"):
                                    log.debug("RETINAL RL: Uploading %s",f)
                                    wandb.log({f: wandb.Video(os.path.join(path,f))})


                        #for f in os.listdir(pltpth):
                        #    if f.endswith(".png"):
                        #        log.debug("RETINAL RL: Uploading %s",f)
                        #        wandb.log({f: wandb.Image(os.path.join(pltpth,f))})
                        ## load all mp4 files in the plot directory and upload them to wandb
                        #for f in os.listdir(pltpth):
                        #    if f.endswith(".mp4"):
                        #        log.debug("RETINAL RL: Uploading %s",f)
                        #        wandb.log({f: wandb.Video(os.path.join(pltpth,f))})

                    self.steps_complete += 1
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

def fill_in_argv_template(argv):
    """Replace string templates in argv with values from argv."""

    # Convert argv into a dictionary
    argv = [a.split('=') for a in argv]
    # Remove dashes from argv
    cfg = dict([[a[0].replace("--",""),a[1]] for a in argv])
    # Replace cfg string templates
    cfg = {k:v.format(**cfg) for k,v in cfg.items()}
    # Convert cfg back into argv
    argv = [f"--{k}={v}" for k,v in cfg.items()]

    return argv



### Main ###


def main():
    """Script entry point."""
    # Register retinal environments and models.

    # Parsing args
    argv = sys.argv[1:]
    # Replace string templates in argv with values from argv.
    argv = fill_in_argv_template(argv)

    # Two-pass building parser and returning cfg : Namespace
    parser,cfg = parse_sf_args(argv,evaluation=True)

    add_retinal_env_args(parser)
    add_retinal_env_eval_args(parser)
    retinal_override_defaults(parser)

    cfg = parse_full_cfg(parser, argv)

    register_retinal_env(cfg)
    register_retinal_model()

    test_env = make_env_func_batched( cfg
            , env_config=AttrDict(worker_index=0, vector_index=0, env_id=0), render_mode="rgb_array"
            )

    obs_space = test_env.observation_space
    enc = make_network(cfg,obs_space).vision_model

    print("Vison Model summary:")
    ts.summary(enc,(3,cfg.res_h,cfg.res_w),receptive_field=True) #,effective_rf_stats=True)
    print("\nEnvironment wrappers:\n")
    # Get string representation of environment wrappers
    print(test_env)

    # Run simulation
    if not(cfg.dry_run):

        status = run_rl(cfg)
        return status

if __name__ == "__main__":
    sys.exit(main())
