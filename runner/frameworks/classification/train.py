"""Training loop for the Brain."""

import logging
import time
from pathlib import Path
from typing import Dict, List

import torch
import wandb
from omegaconf import DictConfig
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader

from retinal_rl.classification.imageset import Imageset
from retinal_rl.classification.loss import ClassificationContext
from retinal_rl.classification.training import process_dataset, run_epoch
from retinal_rl.classification.gpu_transforms import GPUBatchTransforms
from retinal_rl.classification.fast_dataset import FastCIFAR10
from retinal_rl.models.brain import Brain
from retinal_rl.models.objective import Objective
from runner.frameworks.classification.analyze import AnalysesCfg, analyze
from runner.util import save_checkpoint

# Initialize the logger
logger = logging.getLogger(__name__)


def train(
    cfg: DictConfig,
    device: torch.device,
    brain: Brain,
    objective: Objective[ClassificationContext],
    optimizer: Optimizer,
    train_set: Imageset,
    test_set: Imageset,
    initial_epoch: int,
    history: Dict[str, List[float]],
):
    """Train the Brain model using the specified optimizer.

    Args:
    ----
        cfg (DictConfig): The configuration for the experiment.
        device (torch.device): The device to run the computations on.
        brain (Brain): The Brain model to train and evaluate.
        objective (Objective): The optimizer for updating the model parameters.
        train_set (Imageset): The training dataset.
        test_set (Imageset): The test dataset.
        initial_epoch (int): The epoch to start training from.
        history (Dict[str, List[float]]): The training history.

    """

    use_wandb = cfg.logging.use_wandb

    data_dir = Path(cfg.path.data_dir)
    checkpoint_dir = Path(cfg.path.checkpoint_dir)

    max_checkpoints = cfg.logging.max_checkpoints
    checkpoint_step = cfg.logging.checkpoint_step

    num_epochs = cfg.optimizer.num_epochs
    num_workers = cfg.system.num_workers

    # Use configurable batch size with fallback to 64
    batch_size = getattr(cfg, 'batch_size', 64)
    
    # Initialize batch transforms if enabled (but not if using cached transforms)
    use_cached_transforms = getattr(cfg, 'use_cached_transforms', False)
    use_batch_transforms = getattr(cfg, 'use_batch_transforms', False) and not use_cached_transforms
    batch_transforms = None
    if use_batch_transforms:
        from retinal_rl.classification.batch_transforms import BatchGPUTransforms
        # Calculate scale factor range for batch transforms
        # Match the config's scale range or default to fixed scaling
        if hasattr(cfg.dataset.imageset.source_transforms[0], 'image_rescale_range'):
            scale_range = tuple(cfg.dataset.imageset.source_transforms[0].image_rescale_range)
        else:
            # Default to fixed scaling based on vision size
            scale_factor = cfg.vision_width / 32.0  # For 81x81: 81/32 = 2.53
            scale_range = (scale_factor, scale_factor)
        
        batch_transforms = BatchGPUTransforms(
            vision_width=cfg.vision_width,
            vision_height=cfg.vision_height,
            scale_factor_range=scale_range,
            enable_noise=True,  # Enable comprehensive noise transforms
            shot_noise_range=(0.05, 0.1),
            contrast_range=(0.8, 1.2),
            brightness_range=(-0.1, 0.1),
            blur_range=(0.1, 0.5),
            blur_kernel_size=3,
            enable_random_shifts=True  # Enable spatial diversity
        )
    
    # Initialize GPU transforms if enabled
    use_gpu_transforms = getattr(cfg, 'use_gpu_transforms', False)
    gpu_transforms = None
    if use_gpu_transforms:
        gpu_transforms = GPUBatchTransforms(
            target_width=cfg.vision_width,
            target_height=cfg.vision_height,
            scale_factor=2.5
        )
    
    # Check if we need optimized data loaders for batch-native iteration
    use_optimized_live_transforms = getattr(cfg, 'use_optimized_live_transforms', False)
    if use_optimized_live_transforms and hasattr(train_set, 'get_batch_iterator'):
        from retinal_rl.classification.optimized_live_transforms import OptimizedDataLoader
        trainloader = OptimizedDataLoader(train_set, shuffle=True)
        testloader = OptimizedDataLoader(test_set, shuffle=False)
    # Check if we need custom collate function for live vectorized transforms
    elif getattr(cfg, 'use_live_vectorized_transforms', False) and hasattr(train_set, 'collate_fn'):
        trainloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers,
            collate_fn=train_set.collate_fn
        )
        testloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers,
            collate_fn=test_set.collate_fn
        )
    else:
        trainloader = DataLoader(
            train_set, batch_size=batch_size, shuffle=True, num_workers=num_workers
        )
        testloader = DataLoader(
            test_set, batch_size=batch_size, shuffle=False, num_workers=num_workers
        )

    wall_time = time.time()

    # Skip initial evaluation if configured for fast iteration
    skip_initial = getattr(cfg.logging, 'skip_initial_eval', False)
    
    if initial_epoch == 0 and not skip_initial:
        brain.train()
        train_losses = process_dataset(
            device,
            brain,
            objective,
            optimizer,
            initial_epoch,
            trainloader,
            is_training=False,
            gpu_transforms=gpu_transforms,
            batch_transforms=batch_transforms,
        )
        brain.eval()
        test_losses = process_dataset(
            device,
            brain,
            objective,
            optimizer,
            initial_epoch,
            testloader,
            is_training=False,
            gpu_transforms=gpu_transforms,
            batch_transforms=batch_transforms,
        )

        # Initialize the history
        logger.info("Epoch 0 training performance:")
        for key, value in train_losses.items():
            logger.info(f"{key}: {value:.4f}")
            history[f"train_{key}"] = [value]
        for key, value in test_losses.items():
            history[f"test_{key}"] = [value]
    else:
        # Initialize empty history when skipping initial eval
        logger.info("Skipping initial evaluation for faster startup")
        history = {}
        
    # Analysis configuration (common for both paths)
    ana_cfg = AnalysesCfg(
        Path(cfg.path.run_dir),
        Path(cfg.path.plot_dir),
        Path(cfg.path.checkpoint_plot_dir),
        Path(cfg.path.data_dir),
        cfg.logging.use_wandb,
        cfg.logging.channel_analysis,
        cfg.logging.plot_sample_size,
    )
    
    # Only run analysis if not skipping initial eval
    if initial_epoch == 0 and not skip_initial:
        analyze(
            ana_cfg,
            device,
            brain,
            objective,
            history,
            train_set,
            test_set,
            initial_epoch,
            True,
        )

        new_wall_time = time.time()
        epoch_wall_time = new_wall_time - wall_time
        wall_time = new_wall_time
        logger.info(f"Initialization complete. Wall Time: {epoch_wall_time:.2f}s.")

        if use_wandb:
            _wandb_log_statistics(initial_epoch, epoch_wall_time, history)

    else:
        logger.info(
            f"Reloading complete. Resuming training from epoch {initial_epoch}."
        )

    for epoch in range(initial_epoch + 1, num_epochs + 1):
        epoch_start = time.time()
        print(f"=== EPOCH {epoch} DETAILED TIMING ===")
        
        run_epoch_start = time.time()
        brain, history = run_epoch(
            device,
            brain,
            objective,
            optimizer,
            history,
            epoch,
            trainloader,
            testloader,
            gpu_transforms,
            batch_transforms,
        )
        run_epoch_time = time.time() - run_epoch_start
        print(f"run_epoch() call: {run_epoch_time:.3f}s")

        new_wall_time = time.time()
        epoch_wall_time = new_wall_time - wall_time
        wall_time = new_wall_time
        
        epoch_total_time = time.time() - epoch_start
        unaccounted_time = epoch_total_time - run_epoch_time
        print(f"Total epoch time: {epoch_total_time:.3f}s")
        print(f"Unaccounted overhead: {unaccounted_time:.3f}s ({100*unaccounted_time/epoch_total_time:.1f}%)")
        
        logger.info(f"Epoch {epoch} complete. Wall Time: {epoch_wall_time:.2f}s.")

        if epoch % checkpoint_step == 0:
            logger.info("Saving checkpoint and plots.")

            save_checkpoint(
                data_dir,
                checkpoint_dir,
                max_checkpoints,
                brain,
                optimizer,
                history,
                epoch,
            )

            ana_cfg = AnalysesCfg(
                Path(cfg.path.run_dir),
                Path(cfg.path.plot_dir),
                Path(cfg.path.checkpoint_plot_dir),
                Path(cfg.path.data_dir),
                cfg.logging.use_wandb,
                cfg.logging.channel_analysis,
                cfg.logging.plot_sample_size,
            )

            analyze(
                ana_cfg,
                device,
                brain,
                objective,
                history,
                train_set,
                test_set,
                epoch,
                True,
            )
            logger.info("Analysis complete.")

        if use_wandb:
            _wandb_log_statistics(epoch, epoch_wall_time, history)


def _wandb_log_statistics(
    epoch: int, epoch_wall_time: float, histories: Dict[str, List[float]]
) -> None:
    log_dict = {
        "Epoch": epoch,
        "Auxiliary/Epoch Wall Time": epoch_wall_time,
    }

    for key, values in histories.items():
        # Split the key into category (train/test) and metric name
        category, *metric_parts = key.split("_")
        metric_name = " ".join(word.capitalize() for word in metric_parts)

        # Create the full log key
        log_key = f"{category.capitalize()}/{metric_name}"

        # Add to log dictionary
        log_dict[log_key] = values[-1]

    wandb.log(log_dict, commit=True)
