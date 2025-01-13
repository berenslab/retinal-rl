from retinal_rl.analysis.plot import (
    FigureLogger,
    plot_brain_and_optimizers,
    plot_receptive_field_sizes,
)
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import ContextT, Objective
from retinal_rl.util import FloatArray

INIT_DIR = "initialization_analysis"


def initialization_plots(
    log: FigureLogger,
    brain: Brain,
    objective: Objective[ContextT],
    input_shape: tuple[int, ...],
    rf_result: dict[str, FloatArray],
):
    log.save_summary(brain)

    # TODO: This is a bit of a hack, we should refactor this to get the relevant information out of  cnn_stats
    rf_sizes_fig = plot_receptive_field_sizes(input_shape, rf_result)
    log.log_figure(
        rf_sizes_fig,
        INIT_DIR,
        "receptive_field_sizes",
        0,
        False,
    )

    graph_fig = plot_brain_and_optimizers(brain, objective)
    log.log_figure(
        graph_fig,
        INIT_DIR,
        "brain_graph",
        0,
        False,
    )
