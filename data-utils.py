import torch
import torchvision.transforms as transforms
import torchvision.datasets as datasets

def prepare_dataloaders(data_dir, batch_size=64):
    # Define transforms for the training set
    train_transforms = transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Define transforms for the testing set (same as validation)
    test_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Load the datasets with ImageFolder
    train_data = datasets.ImageFolder(data_dir + '/train', transform=train_transforms)
    valid_data = datasets.ImageFolder(data_dir + '/valid', transform=test_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)

    # Create DataLoaders
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(valid_data, batch_size=batch_size)

    return trainloader, validloader, testloader, train_data

def process_image(image_path):
    """
    Scales, crops, and normalizes a PIL image for a PyTorch model,
    returns a Tensor
    """
    from PIL import Image
    
    # Open the image
    pil_img = Image.open(image_path)
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Apply transforms
    tensor_img = transform(pil_img)
    
    return tensor_img

def imshow(image, ax=None, title=None):
    """
    Displays an image with optional axis and title
    """
    import matplotlib.pyplot as plt
    import numpy as np
    
    if ax is None:
        fig, ax = plt.subplots()
    
    # Convert tensor to numpy if needed
    if torch.is_tensor(image):
        image = image.numpy()
    
    # Transpose image
    image = image.transpose((1, 2, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    if title:
        ax.set_title(title)
    
    return ax
