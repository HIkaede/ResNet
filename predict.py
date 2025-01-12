import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
from myres import ResNet18


def main(data_root, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.ToTensor(),
            transforms.Normalize([0.251, 0.234, 0.221], [0.270, 0.254, 0.242]),
        ]
    )

    val_dataset = datasets.ImageFolder(
        root=os.path.join(data_root, "val"), transform=data_transforms
    )
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, num_workers=4)
    print("数据加载成功")

    model = ResNet18(num_classes=5).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()
    print(f"模型加载成功，模型路径: {model_path}")

    all_correct = 0
    with torch.no_grad():
        for i, (images, labels) in enumerate(val_loader):
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            _, preds = outputs.max(1)
            correct = preds.eq(labels).sum().item()
            all_correct += correct
            incorrect = labels.size(0) - correct
            print(
                f"Batch {i + 1}:\t正确 {correct}\t错误 {incorrect}\t正确率 {100 * correct / labels.size(0):.2f}%"
            )
    correct_rate = all_correct / len(val_dataset)
    print(f"总正确率: {100 * correct_rate:.2f}%")


if __name__ == "__main__":
    data_root = "/home/yi/resnet/flower_data"
    model_path = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "resnet18_flower.pth"
    )
    main(data_root=data_root, model_path=model_path)
