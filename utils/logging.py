# utils/logging.py
import logging
from datetime import datetime
import os


def configure_logging(args, dataset_name, time_dir):
    """配置日志"""
    # 保存数据集信息到单独文件
    with open(os.path.join(time_dir, "dataset_info.txt"), "w") as f:
        f.write(f"Dataset: {dataset_name}\n")

    # 配置日志
    log_filename = f"training_{dataset_name}_bs{args.batch_size}_ep{args.n_epochs}_{args.strategy}_th{args.threshold:.3f}_gpu{args.gpu}.log"
    log_filepath = os.path.join(time_dir, log_filename)
    logging.basicConfig(
        filename=log_filepath,
        level=logging.INFO,
        format='%(asctime)s %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M'
    )

    log_all_parameters(args)


def save_args_to_file(args, save_dir):
    """将参数保存到文本文件"""
    args_path = os.path.join(save_dir, "experiment_args.txt")
    with open(args_path, 'w') as f:
        for arg in vars(args):
            f.write(f"{arg}: {getattr(args, arg)}\n")


def log_all_parameters(args):
    """记录所有参数到日志"""
    logging.info("=" * 50)
    logging.info("Experiment Configuration Parameters:")
    for arg in vars(args):
        logging.info(f"{arg}: {getattr(args, arg)}")
    logging.info("=" * 50)
