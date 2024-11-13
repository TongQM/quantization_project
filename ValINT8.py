import torch
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
from tqdm import tqdm

# Import quantization modules
from torch.quantization import get_default_qconfig
from torch.quantization.quantize_fx import prepare_fx, convert_fx

def evaluate_model(model, dataloader, device):
    model.eval()
    correct_top1 = 0
    correct_top5 = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="Evaluating", leave=True):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)

            # Top-1 accuracy
            _, predicted_top1 = outputs.topk(1, dim=1)
            correct_top1 += (predicted_top1.squeeze() == labels).sum().item()
            
            # Top-5 accuracy
            _, predicted_top5 = outputs.topk(5, dim=1)
            correct_top5 += sum([1 if label in pred else 0 for label, pred in zip(labels, predicted_top5)])
            total += labels.size(0)

    top1_acc = 100 * correct_top1 / total
    top5_acc = 100 * correct_top5 / total
    return top1_acc, top5_acc

def main():
    data_dir = ".."  # Update this to your ImageNet validation dataset path
    batch_size = 64
    num_workers = 4  # Adjust based on your system

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    val_dataset = datasets.ImageFolder(root=f"{data_dir}/val", transform=transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )

    # Use CPU or mps device for evaluation; quantization is performed on CPU
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")

    # Define the models to test
    models_to_test = {
        "ResNet18": models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1),
        "MobileNetV2": models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1),
        "ViT_B_16": models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
    }

    for model_name, model in models_to_test.items():
        print(f"\nEvaluating {model_name} (FP32)...")
        model = model.to(device)
        top1_fp32, top5_fp32 = evaluate_model(model, val_loader, device)
        print(f"{model_name} FP32 - Top-1 Accuracy: {top1_fp32:.2f}%, Top-5 Accuracy: {top5_fp32:.2f}%")

        # Quantize the model
        print(f"\nQuantizing {model_name}...")
        model.eval()
        model.cpu()  # Quantization is performed on CPU

        # Get default qconfig based on backend
        torch.backends.quantized.engine = 'qnnpack'
        qconfig = get_default_qconfig('qnnpack')  # Use 'qnnpack' for mobile models or if 'fbgemm' is not available
        qconfig_dict = {"": qconfig}

        # Prepare the model for quantization
        example_inputs = torch.randn(1, 3, 224, 224)
        try:
            prepared_model = prepare_fx(model, qconfig_dict, example_inputs)
        except Exception as e:
            print(f"Failed to prepare {model_name} for quantization: {e}")
            continue

        # Calibrate the model (using a subset of data for speed)
        print(f"Calibrating {model_name}...")
        calibration_batches = 10  # Number of batches to use for calibration
        batch_count = 0
        with torch.no_grad():
            for images, _ in tqdm(val_loader, desc="Calibrating", leave=True):
                prepared_model(images)
                batch_count += 1
                if batch_count >= calibration_batches:
                    break

        # Convert the model to a quantized version
        try:
            quantized_model = convert_fx(prepared_model)
        except Exception as e:
            print(f"Failed to convert {model_name} to quantized model: {e}")
            continue

        quantized_model.to('cpu')

        # Evaluate the quantized model
        print(f"\nEvaluating {model_name} (INT8 Quantized)...")
        top1_int8, top5_int8 = evaluate_model(quantized_model, val_loader, 'cpu')
        print(f"{model_name} INT8 - Top-1 Accuracy: {top1_int8:.2f}%, Top-5 Accuracy: {top5_int8:.2f}%")

if __name__ == "__main__":
    main()
