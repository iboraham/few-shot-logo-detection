import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image
import torch
import json


class LogoDetectionDataset(Dataset):
    """Custom Dataset for loading COCO Images and annotations. Therefore structure of the dataset folder should be,
    - root
        - train
            - _annotations.coco.json
            - img1.png
            - img2.png
        - val
            - _annotations.coco.json
            - img3.png
            - img4.png

    Args:
        root (str): path to dataset directory
        labels_file (str): path to labels file. Defaults to "_annotations.coco.json".
        transforms: data transforms to apply to images. Defaults to None.
    """

    def __init__(
        self,
        data_dir: str,
        labels_file: str = "_annotations.coco.json",
        transforms=None,
    ):
        self.data_dir = data_dir
        self.labels_file = labels_file
        self.transforms = transforms

        # Read the annotations file
        with open(os.path.join(self.data_dir, self.labels_file)) as f:
            self.coco = json.load(f)

        self.image_ids: list = list(
            sorted([image["id"] for image in self.coco["images"]])
        )

    def __getitem__(self, idx: int):
        # Get the image id for the current index
        image_id: str = self.image_ids[idx]

        # Get the image path from the image id
        image_path: str = os.path.join(
            self.data_dir, self.coco["images"][image_id]["file_name"]
        )

        # Open the image
        image: Image.Image = Image.open(image_path).convert("RGB")

        # Get the annotations for the current image
        annotation: list = self.coco["annotations"][image_id]

        # Get bounding box coordinates for each annotation
        box: list = annotation["bbox"]

        # Get class labels for each annotation
        labels: int = annotation["category_id"]

        # Convert everything into a torch.Tensor
        box: torch.Tensor = torch.as_tensor(box, dtype=torch.float32)
        labels: torch.Tensor = torch.as_tensor(labels, dtype=torch.int64)

        # Get the area for bounding box
        logging.error(box)  # tensor([579.0000, 523.0000,  28.5000,  57.0000])
        area: torch.Tensor = (box[2] - box[0]) * (box[3] - box[1])

        # Assume all instances are not crowd
        iscrowd: torch.Tensor = torch.zeros((1,), dtype=torch.int64)

        # Create the target dictionary
        target: dict = {}
        target["boxes"]: torch.Tensor = box
        target["labels"]: torch.Tensor = labels
        target["image_id"]: torch.Tensor = torch.tensor([idx], dtype=torch.int64)
        target["area"]: torch.Tensor = area
        target["iscrowd"]: torch.Tensor = iscrowd

        # Apply transforms on the image
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.image_ids)


if __name__ == "__main__":
    SHOW_SAMPLE_IMAGE = True
    import logging
    from dotenv import load_dotenv

    load_dotenv()
    logging.basicConfig(level=logging.INFO)

    # Load dataset from Roboflow
    logging.info("Loading dataset...")

    if not os.path.exists(os.path.join("data", "game_stats-1")):
        logging.info("Downloading dataset from Roboflow...")
        from roboflow import Roboflow

        rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        project = rf.workspace("dc-bradford-associates").project("game_stats")
        dataset = project.version(1).download("coco")
    else:
        logging.info("Dataset already downloaded, skipping...")

    logging.info("Dataset loaded! Now loading into PyTorch...")

    # Load dataset from local directory
    dataset = LogoDetectionDataset(
        data_dir=os.path.join("data", "game_stats-1", "train"),
    )

    logging.info("Done!")
    logging.info(f"Number of samples: {len(dataset)}")
    logging.info(f"Sample: {dataset[0]}")

    # show sample's image and annotations using pillow
    if SHOW_SAMPLE_IMAGE:
        from PIL import ImageDraw

        img, target = dataset[1]
        draw = ImageDraw.Draw(img)
        # Target boxes are in format [x, y, width, height]
        logging.info(target["boxes"][0])
        draw.rectangle(
            (
                (target["boxes"][0], target["boxes"][1]),
                (
                    target["boxes"][0] + target["boxes"][2],
                    target["boxes"][1] + target["boxes"][3],
                ),
            ),
            outline="red",
            width=3,
        )
        img.show()
