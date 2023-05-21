# this scripts is used to train the model
import yaml
from torch.utils.data import DataLoader, random_split
from utils import collate_fn, get_transforms, seed_everything

from data import LogoDetectionDataset
from model import LogoDetectionModel


def train():
    # Load the configuration file
    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # Set the seed for reproducibility
    seed_everything(config["seed"])

    # Create the dataset
    dataset = LogoDetectionDataset(
        data_dir=config["data_dir"],
        labels_file=config["labels_file"],
        transforms=get_transforms(train=True),
    )

    # Split the dataset into train and validation sets
    train_set, val_set = random_split(
        dataset, [config["train_size"], config["val_size"]]
    )

    # Create the dataloaders
    train_loader = DataLoader(
        train_set,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
        collate_fn=collate_fn,
    )

    # Create the model
    model = LogoDetectionModel(num_classes=config["num_classes"])

    # Move the model to the appropriate device
    model.to(config["device"])
