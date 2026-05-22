from torchvision import transforms
import torch
from torch.utils.data import Dataset
import PIL.Image as Image


class EgocartDataset(Dataset):
    def __init__(self, df, rgb_dir, transform=None):
        self.transform = transform

        # Converting DataFrame columns to numpy arrays for faster access
        self.rgb_paths = (rgb_dir + df["rgb_image_filename"]).to_numpy(copy=True)
        self.labels = (df["c"].astype(int) - 1).to_numpy(copy=True)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        rgb_image = Image.open(self.rgb_paths[idx]).convert("RGB")
        label = self.labels[idx]

        if self.transform:
            rgb_image = self.transform(rgb_image)

        label = torch.tensor(label, dtype=torch.long)

        return rgb_image, label


# Defining transforms for the images
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])
