"""nnU-Net trainer for throughput benchmarks: a few epochs, no checkpoints, no final validation.

setup_env.sbatch copies this file into the installed nnunetv2 package, because nnU-Net only finds
trainer classes that live inside nnunetv2/training/nnUNetTrainer/.

Environment variables:
  NNUNET_BENCH_EPOCHS  number of epochs to run (default 4; the first one includes torch.compile warm-up)
  NNUNET_BENCH_OUT     JSON file to write the timings to (rank 0 writes it; default: in the output folder)
"""
import json
import os
import time

import torch
import torch.distributed as dist

from nnunetv2.training.nnUNetTrainer.nnUNetTrainer import nnUNetTrainer


class nnUNetTrainerBenchmark(nnUNetTrainer):
    # Explicit arguments: nnUNetTrainer records its constructor arguments by name (*args would fail).
    def __init__(self, plans: dict, configuration: str, fold: int, dataset_json: dict,
                 device: torch.device = torch.device("cuda")):
        super().__init__(plans, configuration, fold, dataset_json, device)
        self.num_epochs = int(os.environ.get("NNUNET_BENCH_EPOCHS", 4))
        self._bench = {"epoch_s": [], "train_iter_s": [], "peak_mem_gb": []}
        self._t_epoch = None

    def on_epoch_start(self):
        torch.cuda.reset_peak_memory_stats(self.device)
        torch.cuda.synchronize(self.device)
        self._t_epoch = time.perf_counter()
        super().on_epoch_start()

    def on_train_epoch_end(self, train_outputs):
        torch.cuda.synchronize(self.device)
        self._bench["train_iter_s"].append((time.perf_counter() - self._t_epoch) / self.num_iterations_per_epoch)
        super().on_train_epoch_end(train_outputs)

    def on_epoch_end(self):
        torch.cuda.synchronize(self.device)
        self._bench["epoch_s"].append(time.perf_counter() - self._t_epoch)
        self._bench["peak_mem_gb"].append(torch.cuda.max_memory_allocated(self.device) / 1e9)
        super().on_epoch_end()

    def save_checkpoint(self, filename: str) -> None:
        pass  # benchmark runs keep nothing

    def perform_actual_validation(self, save_probabilities: bool = False):
        self.print_to_log_file("benchmark trainer: skipping the final validation")

    def on_train_end(self):
        super().on_train_end()
        world = dist.get_world_size() if self.is_ddp else 1
        peak = torch.tensor(max(self._bench["peak_mem_gb"]), device=self.device)
        if self.is_ddp:
            dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        if self.local_rank == 0:
            out = os.environ.get("NNUNET_BENCH_OUT") or os.path.join(self.output_folder, "benchmark.json")
            os.makedirs(os.path.dirname(out), exist_ok=True)
            summary = {
                "gpus": world,
                "gpu_name": torch.cuda.get_device_name(self.device),
                "global_batch_size": self.configuration_manager.batch_size,
                "batch_size_per_gpu": self.batch_size,
                "patch_size": list(self.configuration_manager.patch_size),
                "iterations_per_epoch": self.num_iterations_per_epoch,
                "epochs": self.num_epochs,
                "epoch_s": self._bench["epoch_s"],
                "train_iter_s": self._bench["train_iter_s"],
                "peak_mem_gb_per_gpu": float(peak),
                "torch": torch.__version__,
            }
            with open(out, "w") as f:
                json.dump(summary, f, indent=2)
            self.print_to_log_file(f"benchmark summary written to {out}")
