import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from myres import ResNet50
import os
import time


def timer(func):
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        print(f"function {func.__name__} took {time.time()-start:.3f} seconds")
        return result

    return wrapper


@timer
def train_one_epoch(model, dataloader, optimizer, criterion, device):
    model.train()
    running_loss = 0.0
    for images, labels in dataloader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    with torch.no_grad():
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
    return val_loss / len(dataloader), correct / len(dataloader.dataset)


def main(data_root, classes, epochs=20, lr=0.001):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = {
        "train": transforms.Compose(
            [
                transforms.RandomCrop(112),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize([0.251, 0.234, 0.221], [0.270, 0.254, 0.242]),
            ]
        ),
        "val": transforms.Compose(
            [
                transforms.CenterCrop(112),
                transforms.Lambda(lambda img: img.convert("RGB")),
                transforms.ToTensor(),
                transforms.Normalize([0.251, 0.234, 0.221], [0.270, 0.254, 0.242]),
            ]
        ),
    }

    train_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "train"), transform=data_transforms["train"]
    )
    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "val"), transform=data_transforms["val"]
    )

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)

    model = ResNet50(num_classes=classes).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr)

    for epoch in range(epochs):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        print(
            f"Epoch [{epoch+1}/{epochs}], "
            f"Train Loss: {train_loss:.4f}, "
            f"Val Loss: {val_loss:.4f}, "
            f"Val Acc: {val_acc:.4f}"
        )

    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_path = os.path.join(script_dir, "resnet50_pokemon.pth")
    torch.save(model.state_dict(), save_path)


if __name__ == "__main__":
    main(
        data_root="/home/yi/resnet/pokemon-dataset-1000",
        classes=1000,
        epochs=100,
        lr=0.001,
    )
