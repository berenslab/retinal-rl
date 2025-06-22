"""Handles the configuration and initialization of datasets for experiments."""

import os

import hydra
from omegaconf import DictConfig
from torchvision import datasets

from retinal_rl.classification.imageset import Imageset
from retinal_rl.classification.fast_dataset import FastCIFAR10
from retinal_rl.classification.batch_transforms import FastCIFAR10BatchDataset
from retinal_rl.classification.cached_transforms import CachedDatasetManager
from retinal_rl.classification.vectorized_batch_transforms import VectorizedCachedDataset
from retinal_rl.classification.live_vectorized_transforms import create_live_vectorized_datasets
from retinal_rl.classification.optimized_live_transforms import create_optimized_live_datasets


def get_datasets(cfg: DictConfig) -> tuple[Imageset, Imageset]:
    """Get the train and test datasets based on the configuration."""
    cache_dir = os.path.join(hydra.utils.get_original_cwd(), "cache")
    
    # Use optimized live transforms if enabled (batch-native, no individual sample access)
    use_optimized_live_transforms = getattr(cfg, 'use_optimized_live_transforms', False)
    if use_optimized_live_transforms and cfg.dataset.name.upper() == "CIFAR10":
        return _get_optimized_live_datasets(cache_dir, cfg)
    
    # Use live vectorized transforms if enabled (no caching, unlimited diversity)
    use_live_vectorized_transforms = getattr(cfg, 'use_live_vectorized_transforms', False)
    if use_live_vectorized_transforms and cfg.dataset.name.upper() == "CIFAR10":
        return _get_live_vectorized_datasets(cache_dir, cfg)
    
    # Use vectorized cached transforms if enabled (fastest with full transform fidelity)
    use_vectorized_transforms = getattr(cfg, 'use_vectorized_transforms', False)
    if use_vectorized_transforms and cfg.dataset.name.upper() == "CIFAR10":
        cache_size = getattr(cfg, 'cache_size', 500000)
        return _get_vectorized_datasets(cache_dir, cfg, cache_size)
    
    # Use cached transforms if enabled (fast with full transform fidelity)
    use_cached_transforms = getattr(cfg, 'use_cached_transforms', False)
    if use_cached_transforms and cfg.dataset.name.upper() == "CIFAR10":
        cache_size = getattr(cfg, 'cache_size', 500000)
        return _get_cached_datasets(cache_dir, cfg, cache_size)
    
    # Use batch transforms if enabled
    use_batch_transforms = getattr(cfg, 'use_batch_transforms', False)
    if use_batch_transforms and cfg.dataset.name.upper() == "CIFAR10":
        return _get_batch_datasets(cache_dir)
    
    # Use FastCIFAR10 if GPU transforms are enabled  
    use_gpu_transforms = getattr(cfg, 'use_gpu_transforms', False)
    if use_gpu_transforms and cfg.dataset.name.upper() == "CIFAR10":
        return _get_fast_datasets(cache_dir)
    else:
        return _get_datasets(cache_dir, cfg.dataset.name, cfg.dataset.imageset)


def _get_datasets(
    cache_dir: str, dataset_name: str, imageset: DictConfig
) -> tuple[Imageset, Imageset]:
    """Get the train and test datasets based on the configuration."""
    os.makedirs(cache_dir, exist_ok=True)

    # Load the base datasets
    if dataset_name.upper() == "CIFAR10":
        train_base = datasets.CIFAR10(root=cache_dir, train=True, download=True)
        test_base = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    elif dataset_name.upper() == "MNIST":
        train_base = datasets.MNIST(root=cache_dir, train=True, download=True)
        test_base = datasets.MNIST(root=cache_dir, train=False, download=True)
    elif dataset_name.upper() == "SVHN":
        train_base = datasets.SVHN(root=cache_dir, split="train", download=True)
        test_base = datasets.SVHN(root=cache_dir, split="test", download=True)
    elif (
        dataset_name.upper() == "RL_STREAM"
    ):  # TODO: Reconsider if this is the approach to go for
        train_base = datasets.ImageFolder(root=cache_dir)
        test_base = datasets.ImageFolder(root=cache_dir)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    # Instantiate the Imagesets using Hydra
    train_set = hydra.utils.instantiate(imageset, base_dataset=train_base)
    test_set = hydra.utils.instantiate(imageset, base_dataset=test_base)

    return train_set, test_set


def _get_fast_datasets(cache_dir: str) -> tuple[FastCIFAR10, FastCIFAR10]:
    """Get fast CIFAR10 datasets that bypass CPU transforms."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create FastCIFAR10 datasets directly
    train_set = FastCIFAR10(root=cache_dir, train=True, download=True)
    test_set = FastCIFAR10(root=cache_dir, train=False, download=True)
    
    return train_set, test_set


def _get_batch_datasets(cache_dir: str) -> tuple[FastCIFAR10BatchDataset, FastCIFAR10BatchDataset]:
    """Get batch transform datasets that apply transforms at batch level."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Create batch datasets  
    train_set = FastCIFAR10BatchDataset(root=cache_dir, train=True, download=True)
    test_set = FastCIFAR10BatchDataset(root=cache_dir, train=False, download=True)
    
    return train_set, test_set


def _get_cached_datasets(cache_dir: str, cfg: DictConfig, cache_size: int) -> tuple:
    """Get cached transform datasets with pre-generated transformed images."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load base datasets
    train_base = datasets.CIFAR10(root=cache_dir, train=True, download=True)
    test_base = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    
    # Create cached datasets
    train_set, test_set = CachedDatasetManager.create_cached_datasets(
        cfg=cfg,
        base_train_dataset=train_base,
        base_test_dataset=test_base,
        cache_size=cache_size
    )
    
    return train_set, test_set


def _get_vectorized_datasets(cache_dir: str, cfg: DictConfig, cache_size: int) -> tuple:
    """Get vectorized cached datasets with ultra-fast batch transform generation."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Load base datasets
    train_base = datasets.CIFAR10(root=cache_dir, train=True, download=True)
    test_base = datasets.CIFAR10(root=cache_dir, train=False, download=True)
    
    # Extract transform parameters from config
    transform_config = {
        'vision_width': cfg.vision_width,
        'vision_height': cfg.vision_height,
        'scale_factor_range': tuple(cfg.dataset.imageset.source_transforms[0].image_rescale_range),
        'shot_noise_range': (0.05, 0.15),  # Default values
        'contrast_range': (0.7, 1.3),
        'brightness_range': (0.8, 1.2), 
        'blur_range': (0.0, 0.8),
        'blur_kernel_size': 5,
        'enable_random_shifts': True
    }
    
    # Override with config values if available
    if hasattr(cfg.dataset.imageset, 'noise_transforms'):
        for transform in cfg.dataset.imageset.noise_transforms:
            if 'ShotNoiseTransform' in transform._target_:
                if hasattr(transform, 'lambda_range'):
                    transform_config['shot_noise_range'] = tuple(transform.lambda_range)
            elif 'ContrastTransform' in transform._target_:
                if hasattr(transform, 'contrast_range'):
                    transform_config['contrast_range'] = tuple(transform.contrast_range)
            elif 'IlluminationTransform' in transform._target_:
                if hasattr(transform, 'brightness_range'):
                    transform_config['brightness_range'] = tuple(transform.brightness_range)
            elif 'BlurTransform' in transform._target_:
                if hasattr(transform, 'blur_range'):
                    transform_config['blur_range'] = tuple(transform.blur_range)
                if hasattr(transform, 'kernel_size'):
                    transform_config['blur_kernel_size'] = transform.kernel_size
    
    # Split cache size for train/test
    train_cache_size = int(cache_size * 0.8)
    test_cache_size = int(cache_size * 0.2)
    
    print(f"Creating vectorized datasets: train={train_cache_size}, test={test_cache_size}")
    
    # Create vectorized cached datasets
    train_set = VectorizedCachedDataset(
        base_dataset=train_base,
        transform_config=transform_config,
        cache_size=train_cache_size,
        batch_generation_size=1024  # Large batches for efficiency
    )
    
    test_set = VectorizedCachedDataset(
        base_dataset=test_base,
        transform_config=transform_config,
        cache_size=test_cache_size,
        batch_generation_size=1024
    )
    
    print(f"Vectorized datasets created: train={len(train_set)}, test={len(test_set)}")
    
    return train_set, test_set


def _get_live_vectorized_datasets(cache_dir: str, cfg: DictConfig) -> tuple:
    """Get live vectorized datasets with on-the-fly batch transforms (no caching)."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract transform parameters from config
    transform_config = {
        'vision_width': cfg.vision_width,
        'vision_height': cfg.vision_height,
        'scale_factor_range': tuple(cfg.dataset.imageset.source_transforms[0].image_rescale_range),
        'shot_noise_range': (0.05, 0.15),  # Default values
        'contrast_range': (0.7, 1.3),
        'brightness_range': (0.8, 1.2), 
        'blur_range': (0.0, 0.8),
        'blur_kernel_size': 5,
        'enable_random_shifts': True
    }
    
    # Override with config values if available
    if hasattr(cfg.dataset.imageset, 'noise_transforms'):
        for transform in cfg.dataset.imageset.noise_transforms:
            if 'ShotNoiseTransform' in transform._target_:
                if hasattr(transform, 'lambda_range'):
                    transform_config['shot_noise_range'] = tuple(transform.lambda_range)
            elif 'ContrastTransform' in transform._target_:
                if hasattr(transform, 'contrast_range'):
                    transform_config['contrast_range'] = tuple(transform.contrast_range)
            elif 'IlluminationTransform' in transform._target_:
                if hasattr(transform, 'brightness_range'):
                    transform_config['brightness_range'] = tuple(transform.brightness_range)
            elif 'BlurTransform' in transform._target_:
                if hasattr(transform, 'blur_range'):
                    transform_config['blur_range'] = tuple(transform.blur_range)
                if hasattr(transform, 'kernel_size'):
                    transform_config['blur_kernel_size'] = transform.kernel_size
    
    print("Creating live vectorized datasets (no caching, unlimited diversity)...")
    
    # Create live vectorized datasets
    train_set, test_set = create_live_vectorized_datasets(
        cache_dir=cache_dir,
        cfg=cfg,
        transform_config=transform_config
    )
    
    print(f"Live vectorized datasets created: train={len(train_set)}, test={len(test_set)}")
    
    return train_set, test_set


def _get_optimized_live_datasets(cache_dir: str, cfg: DictConfig) -> tuple:
    """Get optimized live datasets with batch-native loading (no individual sample access)."""
    os.makedirs(cache_dir, exist_ok=True)
    
    # Extract transform parameters from config
    transform_config = {
        'vision_width': cfg.vision_width,
        'vision_height': cfg.vision_height,
        'scale_factor_range': tuple(cfg.dataset.imageset.source_transforms[0].image_rescale_range),
        'shot_noise_range': (0.05, 0.15),  # Default values
        'contrast_range': (0.7, 1.3),
        'brightness_range': (0.8, 1.2), 
        'blur_range': (0.0, 0.8),
        'blur_kernel_size': 5,
        'enable_random_shifts': True
    }
    
    # Override with config values if available
    if hasattr(cfg.dataset.imageset, 'noise_transforms'):
        for transform in cfg.dataset.imageset.noise_transforms:
            if 'ShotNoiseTransform' in transform._target_:
                if hasattr(transform, 'lambda_range'):
                    transform_config['shot_noise_range'] = tuple(transform.lambda_range)
            elif 'ContrastTransform' in transform._target_:
                if hasattr(transform, 'contrast_range'):
                    transform_config['contrast_range'] = tuple(transform.contrast_range)
            elif 'IlluminationTransform' in transform._target_:
                if hasattr(transform, 'brightness_range'):
                    transform_config['brightness_range'] = tuple(transform.brightness_range)
            elif 'BlurTransform' in transform._target_:
                if hasattr(transform, 'blur_range'):
                    transform_config['blur_range'] = tuple(transform.blur_range)
                if hasattr(transform, 'kernel_size'):
                    transform_config['blur_kernel_size'] = transform.kernel_size
    
    print("Creating optimized live datasets (batch-native, maximum performance)...")
    
    # Create optimized live datasets
    train_set, test_set = create_optimized_live_datasets(
        cache_dir=cache_dir,
        cfg=cfg,
        transform_config=transform_config
    )
    
    print(f"Optimized live datasets created: train={len(train_set)}, test={len(test_set)}")
    
    return train_set, test_set
