import logging
import os
import sys

import torch
from torch.utils.tensorboard import SummaryWriter

from config.config import Config
from data.dataloaders import (
    get_dataset_tasks,
    get_permuted_mnist_tasks,
    get_task_loader,
)
from training.trainer import (
    create_new_component,
    find_latest_checkpoint,
    freeze_component,
    load_model_state,
    save_model_state,
    train_component,
)
from utils.data_tracker import DataTracker
from utils.logging import configure_logging, save_args_to_file
from utils.memory import DualMemoryBuffer
from utils.testing import test_components


def _load_stream(args):
    if args.dataset.lower() == "permuted_mnist":
        return get_permuted_mnist_tasks(
            num_tasks=args.num_tasks,
            fraction=args.dataset_fraction,
            train_batch_size=args.stream_batch_size,
            test_batch_size=args.test_batch_size,
        )

    return get_dataset_tasks(
        dataset_name=args.dataset,
        num_tasks=args.num_tasks,
        fraction=args.dataset_fraction,
        train_batch_size=args.stream_batch_size,
        test_batch_size=args.test_batch_size,
    )


def _train_on_long_memory(component, memory, args, device):
    memory_samples, memory_labels = memory.get_long_samples()
    if memory_samples.size(0) == 0:
        return

    train_component(
        component,
        memory_samples.to(device),
        memory_labels.to(device),
        args.batch_size,
        args.n_epochs,
    )


def _format_memory_line(stats, args):
    return (
        f"short={stats['short_samples']}/{args.short_memory_size} "
        f"long={stats['long_samples']}/{args.long_memory_size} "
        f"total={stats['total_samples']}/{args.memory_size} "
        f"short_cls={stats['short']['class_distribution']} "
        f"long_cls={stats['long']['class_distribution']}"
    )


def _print_effective_config(args):
    print(
        "Effective config: "
        f"vae=FSVAE, "
        f"n_epochs={args.n_epochs}, "
        f"dataset_fraction={args.dataset_fraction}, "
        f"batch_size={args.batch_size}, "
        f"stream_batch_size={args.stream_batch_size}, "
        f"test_batch_size={args.test_batch_size}, "
        f"classifier={args.classifier_type}, "
        f"short_memory={args.short_memory_size}, "
        f"long_memory={args.long_memory_size}, "
        f"model_dir={args.model_dir}"
    )
    print(f"Command line: {' '.join(sys.argv)}")


def main():
    config = Config()
    args = config.parse_args()
    _print_effective_config(args)

    time_dir = config.generate_log_dir(args, args.dataset)
    configure_logging(args, args.dataset, time_dir)
    save_args_to_file(args, time_dir)
    writer = SummaryWriter(log_dir=time_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logging.info("Using device: %s", device)

    memory = DualMemoryBuffer(
        short_size=args.short_memory_size,
        long_size=args.long_memory_size,
        input_channels=args.input_channels,
        img_size=args.img_size,
    )
    checkpoint_path, resume_stream_id = find_latest_checkpoint(args.model_dir, args.num_tasks)
    if checkpoint_path is None:
        components = []
        component = None
        print(
            f"No unfinished stream checkpoint found in {args.model_dir}; "
            "starting a fresh training run"
        )
        logging.info(
            "No unfinished stream checkpoint found in %s; starting fresh",
            args.model_dir,
        )
    else:
        components, component = load_model_state(checkpoint_path, args)
        print(
            f"Resume checkpoint: {checkpoint_path}. "
            f"Streams 1-{resume_stream_id} are treated as finished; "
            f"next zero-based task_id={resume_stream_id} "
            f"(human stream/task {resume_stream_id + 1})."
        )
        logging.info(
            "Resume checkpoint %s; resume_stream_id=%s",
            checkpoint_path,
            resume_stream_id,
        )
    data_tracker = DataTracker(save_dir=time_dir)

    task_datasets, test_loaders = _load_stream(args)
    expansion_count = len(components)
    data_tracker.expansion_count = expansion_count
    decision_step = 0

    if resume_stream_id >= len(task_datasets):
        print("Checkpoint already covers all configured streams; skipping training loop")
        logging.info("Checkpoint already covers all configured streams")

    for stream_id, stream_dataset in enumerate(
        task_datasets[resume_stream_id:],
        start=resume_stream_id,
    ):
        logging.info("Starting stream segment %s/%s", stream_id + 1, len(task_datasets))
        task_loader = get_task_loader(stream_dataset, batch_size=args.stream_batch_size)

        for batch_idx, (data, labels) in enumerate(task_loader):
            data = data.to(device)
            labels = labels.to(device)
            data_tracker.update_data_flow(data.size(0))

            if len(components) == 0:
                component = create_new_component(args, components)
                expansion_count += 1
                data_tracker.increment_expansion()
                print(f"Initial expansion: created expert #{len(components)}")
                logging.info("Initial expansion created expert #%s", len(components))

            data_cpu = data.detach().cpu()
            labels_cpu = labels.detach().cpu()
            short_ready = False

            if not memory.long_full:
                room = memory.long_remaining
                to_long = min(room, data_cpu.size(0))
                added = memory.add_long_samples(data_cpu[:to_long], labels_cpu[:to_long])

                print(
                    f"[Stream {stream_id + 1} Batch {batch_idx}] "
                    f"added {added} samples to long memory"
                )
                logging.info("Added %s samples to long memory", added)

                if added > 0:
                    _train_on_long_memory(component, memory, args, device)

                if to_long < data_cpu.size(0):
                    short_ready = memory.add_short_samples(
                        data_cpu[to_long:],
                        labels_cpu[to_long:],
                    )
                    print(
                        f"Long memory is full; routed "
                        f"{data_cpu.size(0) - to_long} overflow samples to short memory"
                    )

                stats = memory.get_statistics()
                memory_line = _format_memory_line(stats, args)
                print(
                    f"[Stream {stream_id + 1} Batch {batch_idx}] "
                    f"{memory_line} experts={len(components)}"
                )
                logging.info("Batch %s | %s experts=%s", batch_idx, memory_line, len(components))

                if not memory.long_full or not short_ready:
                    continue
            else:
                short_ready = memory.add_short_samples(data_cpu, labels_cpu)
                stats = memory.get_statistics()
                memory_line = _format_memory_line(stats, args)
                print(
                    f"[Stream {stream_id + 1} Batch {batch_idx}] "
                    f"long memory full; added batch to short memory | "
                    f"{memory_line} experts={len(components)}"
                )
                logging.info("Batch %s | %s experts=%s", batch_idx, memory_line, len(components))

                if not short_ready:
                    print("Short memory is not full yet; skip training until drift check")
                    continue

            distance = memory.compute_fsvae_mmd(
                component,
                batch_size=args.encode_batch_size,
            )
            should_expand = distance > args.threshold

            writer.add_scalar("SDM/fsvae_mmd", distance, decision_step)
            writer.add_scalar("SDM/expert_count", len(components), decision_step)
            writer.add_scalar("SDM/long_memory_size", len(memory.long_buffer), decision_step)
            writer.add_scalar("SDM/short_memory_size", len(memory.short_buffer), decision_step)

            print(
                f"[SDM decision {decision_step}] "
                f"FSVAE-MMD={distance:.6f}, threshold={args.threshold:.6f}, "
                f"expand={should_expand}, experts={len(components)}"
            )
            logging.info(
                "SDM decision %s | FSVAE-MMD=%.6f threshold=%.6f expand=%s experts=%s",
                decision_step,
                distance,
                args.threshold,
                should_expand,
                len(components),
            )
            print("Memory before decision:")
            memory.print_statistics()

            if should_expand:
                expansion_count += 1
                data_tracker.increment_expansion()
                seed_samples, seed_labels = memory.get_short_samples(shuffle=False)

                freeze_component(component)
                component = create_new_component(args, components)
                memory.reset()
                seeded = memory.add_long_samples(seed_samples, seed_labels)

                print(
                    f"Expansion #{expansion_count}: froze previous expert, "
                    f"created expert #{len(components)}, "
                    f"seeded new long memory with {seeded} drift-window samples"
                )
                logging.info(
                    "Expansion #%s | created expert #%s seeded=%s",
                    expansion_count,
                    len(components),
                    seeded,
                )

                if seeded > 0:
                    _train_on_long_memory(component, memory, args, device)

                print("Memory after expansion reset/seed:")
                memory.print_statistics()
            else:
                selected_from_short = memory.consolidate(
                    component,
                    batch_size=args.encode_batch_size,
                )
                memory.clear_short()

                print(
                    f"No expansion: selected {selected_from_short} short-memory samples "
                    f"into long memory by FSVAE ELBO, then cleared short memory"
                )
                logging.info(
                    "No expansion | selected_from_short=%s",
                    selected_from_short,
                )

                _train_on_long_memory(component, memory, args, device)
                print("Memory after consolidation:")
                memory.print_statistics()

            data_tracker.record_state(task_id=stream_id, batch_idx=batch_idx)
            decision_step += 1

        if components:
            save_model_state(components, time_dir, f"stream_{stream_id + 1}_model.pth")
            save_model_state(components, args.model_dir, f"stream_{stream_id + 1}_model.pth")
        logging.info(
            "Finished stream segment %s | expansions=%s experts=%s",
            stream_id + 1,
            expansion_count,
            len(components),
        )

    if components:
        save_model_state(components, time_dir, "model_final.pth")
        save_model_state(components, args.model_dir, "model_final.pth")
        test_components(components, test_loaders, device, args)
    data_tracker.save_records()
    writer.close()

    logging.info("Training finished")


if __name__ == "__main__":
    main()
