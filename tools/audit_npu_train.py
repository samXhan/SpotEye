import logging
import os
import tempfile
import time
from collections import Counter
from pathlib import Path

import torch
import torch_npu
from torch.profiler import record_function
from train_ps_net import Trainer

from psd2.config import get_cfg
from psd2.engine import TrainerBase, default_argument_parser, default_setup, hooks, launch


logger = logging.getLogger("psd2.npu_audit")


def tensors(value):
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from tensors(item)
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from tensors(item)


def elements_by_device(values):
    totals = Counter()
    for value in values:
        totals[f"{value.device}/{value.dtype}"] += value.numel()
    return dict(totals)


def optimizer_devices(optimizer):
    return dict(
        Counter(
            f"{name}/{value.device}"
            for state in optimizer.state.values()
            for name, value in state.items()
            if isinstance(value, torch.Tensor)
        )
    )


def log_first_forward(name, module):
    def log_devices(_, inputs, output):
        handle.remove()
        logger.info(
            "NPU_AUDIT module=%s input_elements=%s output_elements=%s",
            name or "model",
            elements_by_device(tensors(inputs)),
            elements_by_device(tensors(output)),
        )
        if not name and isinstance(output, dict):
            logger.info(
                "NPU_AUDIT losses=%s",
                {
                    key: str(value.device)
                    for key, value in output.items()
                    if isinstance(value, torch.Tensor)
                },
            )

    handle = module.register_forward_hook(log_devices)


class AuditTrainer(Trainer):
    def build_hooks(self):
        return [hooks.LRScheduler()]

    def run_step(self):
        relative_iter = self.iter - self.start_iter
        if relative_iter < self.audit_warmup:
            super().run_step()
            if relative_iter + 1 == self.audit_warmup:
                torch.npu.synchronize()
        else:
            start = time.perf_counter()
            with record_function(f"SpotEye::train_step/{self.iter}"):
                super().run_step()
                torch.npu.synchronize()
            logger.info(
                "NPU_AUDIT iter=%d wall_sec=%.3f data_sec=%.3f",
                self.iter,
                time.perf_counter() - start,
                self.storage.history("data_time").latest(),
            )
        self.audit_profiler.step()


def audit(args):
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts or [])
    checkpoint_dir = cfg.OUTPUT_DIR
    output_root = Path(args.audit_dir).resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    output_dir = Path(tempfile.mkdtemp(prefix="run-", dir=output_root))
    cfg.OUTPUT_DIR = str(output_dir)
    cfg.freeze()
    default_setup(cfg, args)

    trainer = AuditTrainer(cfg)
    trainer.checkpointer.save_to_disk = False
    if args.resume:
        trainer.checkpointer.save_dir = checkpoint_dir
        if not trainer.checkpointer.has_checkpoint():
            raise FileNotFoundError(f"No last_checkpoint in {checkpoint_dir}")
    trainer.resume_or_load(resume=args.resume)
    model = trainer.model
    logger.info(
        "NPU_AUDIT model=%s configured_device=%s current_npu=%d batch_size=%d amp=%s "
        "torch=%s torch_npu=%s",
        type(model).__name__,
        cfg.MODEL.DEVICE,
        torch.npu.current_device(),
        trainer.cfg.SOLVER.IMS_PER_BATCH,
        cfg.SOLVER.AMP.ENABLED,
        torch.__version__,
        torch_npu.__version__,
    )
    logger.info(
        "NPU_AUDIT runtime_env=%s",
        {
            name: os.environ.get(name)
            for name in ("ASCEND_LAUNCH_BLOCKING", "TASK_QUEUE_ENABLE", "TORCH_NPU_DISABLED_WARNING")
        },
    )
    logger.info(
        "NPU_AUDIT parameter_elements=%s trainable_elements=%s buffer_elements=%s "
        "optimizer=%s optimizer_state_before=%s",
        elements_by_device(model.parameters()),
        elements_by_device(p for p in model.parameters() if p.requires_grad),
        elements_by_device(model.buffers()),
        type(trainer.optimizer).__name__,
        optimizer_devices(trainer.optimizer),
    )
    traced_modules = {
        "",
        "clip_model.visual.conv1",
        "clip_model.visual.layer3",
        "proposal_generator",
        "roi_heads.pooler.level_poolers.0",
        "roi_heads.res5",
        "clip_model.visual.attnpool",
        "clip_model.transformer",
        "cross_modal_transformer",
        "oim_loss",
        "mim_head",
        "depth_enc3",
        "depth_enc4",
        "depth_pooler.level_poolers.0",
    }
    for name, module in model.named_modules():
        if name in traced_modules:
            log_first_forward(name, module)

    trainer.audit_warmup = args.audit_warmup
    profiler = torch_npu.profiler
    trainer.audit_profiler = profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(
            wait=args.audit_warmup - 1,
            warmup=1,
            active=args.audit_steps,
            repeat=1,
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(str(output_dir)),
        experimental_config=profiler._ExperimentalConfig(
            export_type=profiler.ExportType.Text,
            profiler_level=profiler.ProfilerLevel.Level1,
        ),
        record_shapes=True,
        with_stack=True,
    )
    stop_iter = trainer.start_iter + args.audit_warmup + args.audit_steps
    with trainer.audit_profiler:
        TrainerBase.train(trainer, trainer.start_iter, stop_iter)

    missing_gradients = Counter(
        ".".join(name.split(".")[:3])
        for name, param in model.named_parameters()
        if param.requires_grad and param.grad is None
    )
    logger.info(
        "NPU_AUDIT gradient_elements=%s missing_gradients=%s optimizer_state_after=%s",
        elements_by_device(p.grad for p in model.parameters() if p.grad is not None),
        dict(missing_gradients),
        optimizer_devices(trainer.optimizer),
    )
    logger.info("NPU_AUDIT output=%s", output_dir)
    if not any(output_dir.rglob("trace_view.json")):
        raise RuntimeError(f"Ascend profiler did not produce trace_view.json in {output_dir}")


if __name__ == "__main__":
    parser = default_argument_parser()
    parser.add_argument("--audit-dir", default="outputs/npu_audit")
    parser.add_argument("--audit-warmup", type=int, default=3)
    parser.add_argument("--audit-steps", type=int, default=3)
    args = parser.parse_args()
    if args.eval_only or args.num_machines != 1 or args.num_gpus != [1] or args.machine_rank != 0:
        parser.error("The training audit requires one machine and one NPU.")
    if not args.config_file or min(args.audit_warmup, args.audit_steps) < 1:
        parser.error("Provide --config-file and positive warmup and audit step counts.")
    launch(audit, args.num_gpus, args=(args,))
