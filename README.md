# GTSRB-Dataloader

Loads the official GTSRB (German Traffic Sign Recognition) training and test sets found here:
http://benchmark.ini.rub.de/?section=gtsrb&subsection=dataset

Tested with PyTorch 1.0

Modifications I made to the original dataset to simplify things:

- Training Set - Created one CSV file containing all image paths and classes
- Test Set - Created one CSV file containing all image paths and classes

## Setup
 
- Download GTSRB.zip from OneDrive https://1drv.ms/u/s!An8jrZtDgrMljdt7o2khe7TGmZWbUg
- Unzip
- Use gtsrb_dataset.py to load dataset

## Usage

Create your PyTorch transforms as required

```
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize((0.3403, 0.3121, 0.3214),
                         (0.2724, 0.2608, 0.2669))
])
```
Create Dataset
```
trainset = dataset.GTSRB(
    root_dir='./data', train=True,  transform=transform)
testset = dataset.GTSRB(
    root_dir='./data', train=False,  transform=transform)
```
Use dataset with PyTorch Dataloader
```
trainloader = torch.utils.data.DataLoader(
    trainset, batch_size=128, shuffle=True, num_workers=2)
testloader = torch.utils.data.DataLoader(
    testset, batch_size=128, shuffle=False, num_workers=2)
```
