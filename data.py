import os
import pandas as pd
from torch.utils.data import Dataset
from PIL import Image


class LogoDetectionDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.labels = pd.read_csv(labels_file)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        image_name = self.labels.iloc[idx]["filename"]
        image_path = os.path.join(self.data_dir, image_name)
        image = Image.open(image_path)

        brand_name = self.labels.iloc[idx]["Brand Name"]
        x = self.labels.iloc[idx]["x"]
        y = self.labels.iloc[idx]["y"]
        width = self.labels.iloc[idx]["width"]
        height = self.labels.iloc[idx]["height"]
        image_width = self.labels.iloc[idx]["image_width"]
        image_height = self.labels.iloc[idx]["image_height"]

        sample = {"image": image, "brand_name": brand_name, "x": x, "y": y,
                  "width": width, "height": height, "image_width": image_width,
                  "image_height": image_height}

        if self.transform:
            sample = self.transform(sample)

        return sample


if __name__ == '__main__':
    import logging
    dataset = LogoDetectionDataset(
        data_dir="data/training data", labels_file="data/labels_logo-detection.csv")
    logging.info("Done!")
