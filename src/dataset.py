import os
import random

from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import transforms


class RealAIArtDataset(Dataset):
    """Dataset for loading real and AI-generated artwork images.

    Expects a directory structure where subfolders starting with "AI_"
    contain AI-generated images (label 0) and all other subfolders
    contain real artwork (label 1).
    """

    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        for subfolder in os.listdir(root_dir):
            folder_path = os.path.join(root_dir, subfolder)
            if not os.path.isdir(folder_path):
                continue

            label = 0 if subfolder.startswith("AI_") else 1

            for file in os.listdir(folder_path):
                if file.lower().endswith((".jpg", ".jpeg", ".png")):
                    self.image_paths.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


def get_data_loaders(train_dir, test_dir, batch_size=64, seed=42):
    """Create balanced train/val/test data loaders.

    The training set is balanced by undersampling the larger class to achieve
    a 50/50 split between AI-generated and real images, then split 80/20 into
    training and validation sets.

    Args:
        train_dir: Path to training data directory.
        test_dir: Path to test data directory.
        batch_size: Batch size for all loaders.
        seed: Random seed for reproducibility.

    Returns:
        Tuple of (train_loader, val_loader, test_loader).
    """
    random.seed(seed)

    train_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    eval_transforms = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Load full dataset to get label indices
    full_dataset = RealAIArtDataset(train_dir)

    ai_indices = [i for i, label in enumerate(full_dataset.labels) if label == 0]
    real_indices = [i for i, label in enumerate(full_dataset.labels) if label == 1]

    # Balance by undersampling the larger class
    min_len = min(len(ai_indices), len(real_indices))
    ai_sampled = random.sample(ai_indices, min_len)
    real_sampled = random.sample(real_indices, min_len)
    print(f"Balanced dataset: {len(ai_sampled)} AI + {len(real_sampled)} Real = {min_len * 2} total")

    balanced_indices = ai_sampled + real_sampled
    random.shuffle(balanced_indices)

    # 80/20 train/val split
    train_size = int(0.8 * len(balanced_indices))
    train_indices = balanced_indices[:train_size]
    val_indices = balanced_indices[train_size:]

    train_dataset = Subset(RealAIArtDataset(train_dir, transform=train_transforms), train_indices)
    val_dataset = Subset(RealAIArtDataset(train_dir, transform=eval_transforms), val_indices)
    test_dataset = RealAIArtDataset(test_dir, transform=eval_transforms)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
