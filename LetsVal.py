import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

def evaluate_model(model, dataloader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1准确率
            _, predicted_top1 = outputs.topk(1, dim=1)
            correct_top1 += (predicted_top1.squeeze() == labels).sum().item()
            
            # Top-5准确率
            _, predicted_top5 = outputs.topk(5, dim=1)
            correct_top5 += sum([1 if label in pred else 0 for label, pred in zip(labels, predicted_top5)])
            total += labels.size(0)

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    return top1_acc, top5_acc

def main():
    data_dir = "/Users/zhihanli/Desktop/HW3"
    batch_size = 64
    num_workers = 0

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # 定义要测试的模型
    models_to_test = {
        "ResNet18": models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        "ViT_B_16": models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)  # 标准 ViT-B/16
    }

    for model_name, model in models_to_test.items():
        print(f"\nEvaluating {model_name} (FP32)...")
        model = model.to(device)
        top1, top5 = evaluate_model(model, val_loader, device)
        print(f"{model_name} - Top-1 Accuracy: {top1:.2f}%, Top-5 Accuracy: {top5:.2f}%")

if __name__ == "__main__":
    main()
