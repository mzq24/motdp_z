#!/usr/bin/env python3

import argparse
import copy
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from policy.nuplan_diffusion_policy import NuPlanDiffusionPolicy
from training.train_nuplan import build_eval_dataloader, load_config, validate


DEFAULT_TASK_NAMES = [
    'stationary_in_traffic',
    'starting_straight_traffic_light_intersection_traversal',
    'high_magnitude_speed',
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_overall_eval_dataloader(config: dict):
    return build_eval_dataloader(
        config,
        cache_dirs=config.get('val_cache_dirs', config.get('cache_dirs', [])),
        allowed_scenario_types=config.get(
            'val_allowed_scenario_types',
            config.get('allowed_scenario_types', None),
        ),
        allowed_target_types=config.get(
            'val_allowed_target_types',
            config.get('allowed_target_types', None),
        ),
        max_samples=config.get('val_max_samples', None),
        samples_per_target_type=config.get('val_samples_per_target_type', None),
        repeat_small_target_types=config.get('val_repeat_small_target_types', False),
        sampling_seed=config.get('val_sampling_seed', config.get('sampling_seed', 0)),
        default_target_type_partition_index=config.get(
            'val_default_target_type_partition_index',
            0,
        ),
        target_type_partition_indices=config.get('val_target_type_partition_indices', None),
    )


def build_task_eval_dataloader(config: dict, task_name: str):
    return build_eval_dataloader(
        config,
        cache_dirs=config.get('val_cache_dirs', config.get('cache_dirs', [])),
        allowed_scenario_types=config.get(
            'val_allowed_scenario_types',
            config.get('allowed_scenario_types', None),
        ),
        allowed_target_types=[task_name],
        max_samples=config.get('val_task_max_samples', config.get('val_max_samples', None)),
        samples_per_target_type=config.get('val_task_samples_per_target_type', None),
        repeat_small_target_types=config.get('val_repeat_small_target_types', False),
        sampling_seed=config.get('val_sampling_seed', config.get('sampling_seed', 0)),
        default_target_type_partition_index=config.get(
            'val_default_target_type_partition_index',
            0,
        ),
        target_type_partition_indices=config.get('val_target_type_partition_indices', None),
    )


def configure_policy_norm(policy: NuPlanDiffusionPolicy, config: dict, mode: str) -> None:
    if mode == 'with_norm':
        norm_stats_path = config.get('norm_stats_path')
        if not norm_stats_path:
            raise ValueError('with_norm mode requires norm_stats_path in config')
        with open(norm_stats_path, 'r', encoding='utf-8') as file:
            policy.load_norm_stats(json.load(file))
        return

    if mode == 'identity_norm':
        with torch.no_grad():
            policy.global_abs_mean.zero_()
            policy.global_abs_std.fill_(1.0)
        return

    raise ValueError(f'Unsupported mode: {mode}')


def load_policy(config: dict, checkpoint_path: Path, mode: str) -> dict:
    policy = NuPlanDiffusionPolicy(config).cuda()
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    state_dict = checkpoint.get('model', checkpoint)
    missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
    configure_policy_norm(policy, config, mode)
    policy.eval()
    return {
        'policy': policy,
        'missing_keys': list(missing_keys),
        'unexpected_keys': list(unexpected_keys),
    }


def evaluate_mode(
    base_config: dict,
    checkpoint_path: Path,
    task_names,
    mode: str,
    base_seed: int,
) -> dict:
    config = copy.deepcopy(base_config)
    config['use_tqdm'] = False
    config['num_workers'] = 0
    config['val_num_workers'] = 0

    loaded = load_policy(config, checkpoint_path, mode)
    policy = loaded['policy']

    _, overall_dataloader = build_overall_eval_dataloader(config)
    set_seed(base_seed)
    overall_metrics = validate(
        policy,
        overall_dataloader,
        rank=0,
        config=config,
        progress_desc=f'{mode}:overall',
        max_batches=config.get('val_num_batches', 20),
    )

    task_metrics = {}
    max_task_batches = config.get('val_task_num_batches', config.get('val_num_batches', 20))
    for index, task_name in enumerate(task_names, start=1):
        _, task_dataloader = build_task_eval_dataloader(config, task_name)
        set_seed(base_seed + index)
        task_metrics[task_name] = validate(
            policy,
            task_dataloader,
            rank=0,
            config=config,
            progress_desc=f'{mode}:{task_name}',
            max_batches=max_task_batches,
        )

    return {
        'mode': mode,
        'checkpoint_path': str(checkpoint_path),
        'norm_stats_path': config.get('norm_stats_path'),
        'load_state': {
            'missing_keys': loaded['missing_keys'],
            'unexpected_keys': loaded['unexpected_keys'],
        },
        'overall': overall_metrics,
        'tasks': task_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description='Evaluate a diffusion checkpoint with and without norm stats.')
    parser.add_argument('--config', required=True, help='Path to YAML config')
    parser.add_argument('--checkpoint', required=True, help='Path to checkpoint .pth')
    parser.add_argument('--seed', type=int, default=42, help='Base seed for validation sampling')
    parser.add_argument(
        '--task-name',
        action='append',
        dest='task_names',
        default=None,
        help='Task name to validate; may be passed multiple times',
    )
    parser.add_argument('--output-json', default=None, help='Optional path to save the JSON result')
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this evaluator')

    torch.cuda.set_device(0)
    config = load_config(args.config)
    checkpoint_path = Path(args.checkpoint).expanduser().resolve()
    task_names = args.task_names or config.get('val_task_metric_types') or DEFAULT_TASK_NAMES

    results = {
        'with_norm': evaluate_mode(config, checkpoint_path, task_names, 'with_norm', args.seed),
        'identity_norm': evaluate_mode(config, checkpoint_path, task_names, 'identity_norm', args.seed),
    }

    payload = json.dumps(results, indent=2, sort_keys=True)
    if args.output_json:
        output_path = Path(args.output_json).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(payload + '\n', encoding='utf-8')

    print(payload)


if __name__ == '__main__':
    main()