import torch
from modules import temp
from torchvision import datasets, transforms
from DataInstance import *
from torchattacks import FGSM


def TestRobustness(model, test_dir=r"miniImageNet_d/test"):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    test_dataset = ImageNet.ImageDataset(test_dir, transform=transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=64,
        shuffle=False,
    )

    total = 0
    correct_top1 = 0
    correct_top5 = 0
    correct_top1 = 0
    correct_top5 = 0
    attack = FGSM(model, eps=0.03)
    with torch.no_grad():
        for images, labels in test_loader:
            # print(images)
            images = add_gaussian_noise(images, 0.01)
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted_top1 = torch.max(outputs.data, 1)
            _, predicted_top5 = torch.topk(outputs.data, 5, dim=1)

            total += labels.size(0)
            correct_top1 += (predicted_top1 == labels).sum().item()
            correct_top5 += torch.sum(predicted_top5.eq(labels.view(-1, 1))).item()

    accuracy_top1 = 100 * correct_top1 / total
    accuracy_top5 = 100 * correct_top5 / total

    print(f'Test Accuracy_top1: {accuracy_top1:.2f}% \t Test Accuracy_top5: {accuracy_top5:.2f}%')

def add_gaussian_noise(image, noise_level=0.1):
    noise = torch.randn_like(image) * noise_level
    noise = torch.clamp(noise, -1, 1)
    noisy_image = image + noise
    return noisy_image

model = temp.NlNet(num_classes=100, in_channels=3)
checkpoint = torch.load(r'./MiniImageNetResult/model_test.pt', map_location='cpu', weights_only=True)
model.load_state_dict(checkpoint)
model.eval()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

TestRobustness(model)


