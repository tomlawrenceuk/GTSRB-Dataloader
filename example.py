import gtsrb_dataset as dataset
import torchvision.transforms as transforms
import torch

# Create Transforms
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
])

# Create Datasets
trainset = dataset.GTSRB(
    root_dir='./data', train=True,  transform=transform)
testset = dataset.GTSRB(
    root_dir='./data', train=False,  transform=transform)

# Load Datasets
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)
