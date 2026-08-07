#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generate Mamba benchmark configurations and submit their SLURM jobs."""

import argparse
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Literal

import yaml

BENCHMARK_SUBMIT_SCRIPT = Path(
    "/lustre/fsw/coreai_comparch_trtllm/deemod/TensorRT-LLM/"
    "examples/disaggregated/slurm/benchmark/submit.py"
)
WORKER_ROLES = ("ctx", "gen")

Mode = Literal["v1", "v2", "old_v2"]
Config = dict[str, object]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate benchmark YAML files from every *.yaml in --yaml-dir, "
            "then submit each generated configuration."
        )
    )
    parser.add_argument(
        "--dataset-file",
        "--dataset_file",
        dest="dataset_file",
        required=True,
        help="Dataset path written to benchmark.dataset_file.",
    )
    parser.add_argument(
        "--trtllm-wheel-path",
        "--trtllm_wheel_path",
        dest="trtllm_wheel_path",
        required=True,
        help="Wheel path written to environment.trtllm_wheel_path.",
    )
    parser.add_argument(
        "--log-dir",
        "--log_dir",
        dest="log_dir",
        type=Path,
        required=True,
        help="Directory for generated YAML files and per-job log directories.",
    )
    parser.add_argument(
        "--yaml-dir",
        "--yaml_dir",
        dest="yaml_dir",
        type=Path,
        required=True,
        help="Directory containing source *.yaml files.",
    )

    mode_group = parser.add_mutually_exclusive_group()
    mode_group.add_argument(
        "--v1",
        dest="mode",
        action="store_const",
        const="v1",
        help="Use the C++ transceiver and disable KV cache manager V2.",
    )
    mode_group.add_argument(
        "--v2",
        dest="mode",
        action="store_const",
        const="v2",
        help="Enable KV cache manager V2.",
    )
    mode_group.add_argument(
        "--old-v2",
        "--old_v2",
        dest="mode",
        action="store_const",
        const="old_v2",
        help="Disable KV cache manager V2 without changing the transceiver runtime.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Generate YAML files and print submission commands without executing them.",
    )
    return parser.parse_args()


def require_mapping(parent: Config, key: str, location: str) -> Config:
    value = parent.get(key)
    if not isinstance(value, dict):
        raise ValueError(f"{location}.{key} must be a mapping")
    return value


def load_config(config_path: Path) -> Config:
    with config_path.open(encoding="utf-8") as config_file:
        config = yaml.safe_load(config_file)
    if not isinstance(config, dict):
        raise ValueError(f"{config_path} must contain a YAML mapping")
    return config


def update_config(
    config: Config,
    dataset_file: str,
    trtllm_wheel_path: str,
    mode: Mode | None,
) -> None:
    benchmark_config = require_mapping(config, "benchmark", "config")
    benchmark_config["dataset_file"] = dataset_file

    environment_config = require_mapping(config, "environment", "config")
    environment_config["trtllm_wheel_path"] = trtllm_wheel_path

    if mode is None:
        return

    worker_config = require_mapping(config, "worker_config", "config")
    for role in WORKER_ROLES:
        role_config = require_mapping(worker_config, role, "config.worker_config")
        kv_cache_config = require_mapping(
            role_config,
            "kv_cache_config",
            f"config.worker_config.{role}",
        )
        if mode == "v1":
            transceiver_config = require_mapping(
                role_config,
                "cache_transceiver_config",
                f"config.worker_config.{role}",
            )
            transceiver_config["transceiver_runtime"] = "CPP"
            kv_cache_config["use_kv_cache_manager_v2"] = False
        elif mode == "old_v2":
            kv_cache_config["use_kv_cache_manager_v2"] = False
        else:
            kv_cache_config["use_kv_cache_manager_v2"] = True
            transceiver_config = require_mapping(
                role_config,
                "cache_transceiver_config",
                f"config.worker_config.{role}",
            )
            transceiver_config["kv_cache_bounce_size_mb"] = 512


def save_config(config: Config, output_path: Path) -> None:
    with output_path.open("w", encoding="utf-8") as output_file:
        yaml.safe_dump(
            config,
            output_file,
            default_flow_style=False,
            sort_keys=False,
        )


def generate_configs(
    yaml_dir: Path,
    log_dir: Path,
    dataset_file: str,
    trtllm_wheel_path: str,
    mode: Mode | None,
) -> list[Path]:
    yaml_dir = yaml_dir.expanduser().resolve()
    log_dir = log_dir.expanduser().resolve()

    if not yaml_dir.is_dir():
        raise ValueError(f"YAML directory does not exist: {yaml_dir}")

    source_configs = sorted(yaml_dir.glob("*.yaml"))
    if not source_configs:
        raise ValueError(f"No *.yaml files found in: {yaml_dir}")

    log_dir.mkdir(parents=True, exist_ok=True)
    generated_configs = []
    for source_path in source_configs:
        output_path = log_dir / source_path.name
        if output_path == source_path:
            raise ValueError(f"--log-dir must not overwrite source configuration: {source_path}")

        config = load_config(source_path)
        update_config(config, dataset_file, trtllm_wheel_path, mode)
        save_config(config, output_path)
        generated_configs.append(output_path)
        print(f"Generated: {output_path}")

    return generated_configs


def submit_configs(generated_configs: list[Path], log_dir: Path, dry_run: bool) -> None:
    log_dir = log_dir.expanduser().resolve()
    for config_path in generated_configs:
        job_log_dir = log_dir / config_path.stem
        command = [
            "python3",
            str(BENCHMARK_SUBMIT_SCRIPT),
            "-c",
            str(config_path),
            "--log-dir",
            str(job_log_dir),
        ]
        print(f"Submit: {shlex.join(command)}", flush=True)
        if not dry_run:
            subprocess.run(command, check=True)


def main() -> int:
    args = parse_args()
    try:
        generated_configs = generate_configs(
            yaml_dir=args.yaml_dir,
            log_dir=args.log_dir,
            dataset_file=args.dataset_file,
            trtllm_wheel_path=args.trtllm_wheel_path,
            mode=args.mode,
        )
        submit_configs(generated_configs, args.log_dir, args.dry_run)
    except (OSError, ValueError, yaml.YAMLError, subprocess.CalledProcessError) as error:
        print(f"Error: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
