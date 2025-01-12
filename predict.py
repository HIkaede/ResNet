import torch
import torch.nn as nn
from torchvision import transforms
import os
from myres import ResNet50
from PIL import Image


def main(image_path, model_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transforms = transforms.Compose(
        [
            transforms.Resize(128),
            transforms.CenterCrop(112),
            transforms.Lambda(lambda img: img.convert("RGB")),
            transforms.ToTensor(),
            transforms.Normalize([0.251, 0.234, 0.221], [0.270, 0.254, 0.242]),
        ]
    )

    model = ResNet50(num_classes=1000).to(device)
    model.load_state_dict(
        torch.load(model_path, map_location=device, weights_only=True)
    )
    model.eval()

    try:
        pil_img = Image.open(image_path).convert("RGB")
    except IOError:
        print(f"无法读取图像: {image_path}")
        return

    input_tensor = data_transforms(pil_img).unsqueeze(0).to(device)

    classes_dir = os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "pokemon-dataset-1000/dataset"
    )
    classes = sorted(os.listdir(classes_dir))

    with torch.no_grad():
        output = model(input_tensor)
        _, pred = output.max(1)
        print(f"预测结果: {pred.item()}, {classes[pred.item()]}")


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_model_path = os.path.join(script_dir, "resnet50_pokemon.pth")
    image_path = "/home/yi/resnet/image.png"
    main(image_path=image_path, model_path=default_model_path)
