import os
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUBLAS_WORKSPACE_CONFIG'] = ":4096:8"
import sys
import torch
import random
import numpy as np
import yaml
from torch.utils.data import DataLoader
from datasets.transforms.build import reverse_transforms
from models.best.Poseidon import Poseidon
from datasets.zoo.posetrack.PoseTrack import PoseTrack
from posetimation import get_cfg, update_config
from engine.defaults import default_parse_args
from core.loss import get_loss_function
from core.function import validate
from utils.utils_save_results import save_results
from utils.common import TRAIN_PHASE, VAL_PHASE, TEST_PHASE
import wandb

sys.path.insert(0, os.path.abspath('../'))

def load_config(config_path):
    """ Load the YAML configuration file.

    Args:
        config_path (str): Path to the YAML configuration file.

    Returns:
        dict: Configuration dictionary.
    """
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup(args):
    cfg = get_cfg(args)
    update_config(cfg, args)
    return cfg

def set_seed(config):
    # set the seed for reproducibility
    torch.manual_seed(config.SEED)
    torch.cuda.manual_seed(config.SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(config.SEED)
    random.seed(config.SEED)

def main():
    # Load configuration from a YAML file.
    args = default_parse_args()
    cfg = setup(args)
    
    # Set the seed
    set_seed(cfg)

    # Initialize wandb run if needed
    if cfg.SAVE_RESULTS:
        wandb.init(
            project="poseidon",
            entity="krahim04",
            config=cfg,
            name=cfg.NAME_EXP
        )

    # Set the device
    device = "cuda:" + str(cfg.GPUS[0]) if torch.cuda.is_available() else 'cpu'
    print("\033[92m" + "Device: " + "\033[0m", device)

    # Load the model
    if cfg.MODEL.METHOD == 'poseidon':
        model = Poseidon(cfg, phase=VAL_PHASE, device=device).to(device)

    model_weights_path = args.weights_path
    #model_weights_path = "/home/pace/Poseidon/results/best_results/2024-10-18_2_vitS/best_model.pt"
    #model_weights_path = "/home/pace/Poseidon/results/best_results/2024-10-17_1_VitH/best_model.pt"

    # Assuming model_weights_path is a string that contains the path to the checkpoint file
    # Load the checkpoint onto the CPU and then move it to the GPU
    checkpoint = torch.load(model_weights_path, map_location='cpu')
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)

    # Define loss function (criterion)
    loss = get_loss_function(cfg, device)

    # Load the validation dataset
    val_dataset = PoseTrack(cfg, phase=VAL_PHASE)

    # check if the datasets are loaded successfully
    print("\033[92m" + "Datasets loaded successfully." + "\033[0m")

    # Create the validation DataLoader
    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg.VAL.BATCH_SIZE,
        shuffle=False,
        num_workers=cfg.WORKERS,
        pin_memory=True,
    )

    print("\033[92m" + "Val loader loaded successfully." + "\033[0m")
    print("Number of elements in the val set: ", len(val_dataset))

    # Start the validation process
    print("\033[92m" + "Starting validation..." + "\033[0m")

    performance_values, perf_indicator, val_loss, val_acc = validate(
        cfg, val_loader, val_dataset, model, loss, cfg.OUTPUT_DIR, 999, device=device
    )

    # Log validation results if needed
    if cfg.SAVE_RESULTS:
        # Save the validation results
        results = {
            'performance_values': performance_values,
            'perf_indicator': perf_indicator,
            'loss': val_loss,
        }
        save_results(cfg, results, cfg.OUTPUT_DIR)

        # Log results to wandb
        wandb.log({
            "val_loss": val_loss,
            "val_acc": val_acc,
            "mAP": perf_indicator
        })

    print("Validation completed!")
    print("Performance Indicator (mAP): ", perf_indicator)
    print("Validation Loss: ", val_loss)
    print("Validation Accuracy: ", val_acc)

if __name__ == '__main__':
    # Print environment versioning
    print("\033[94m" + "Environment versioning:" + "\033[0m")

    # Print Python version
    print("Python version: ", sys.version)

    # Print PyTorch version
    print("PyTorch version: ", torch.__version__)

    # Print numpy version
    print("Numpy version: ", np.__version__)

    print("\n\n")
    
    main()