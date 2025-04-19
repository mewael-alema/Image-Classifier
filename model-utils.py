import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
from collections import OrderedDict

def nn_setup(model_name='vgg16', lr=0.001, hidden_units=4096, num_classes=102):
    if model_name == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = 25088
    elif model_name == 'resnet50':
        model = models.resnet50(pretrained=True)
        input_size = model.fc.in_features
    elif model_name == 'vgg13':
        model = models.vgg13(pretrained=True)
        input_size = 25088
    else:
        raise ValueError(f"Unsupported architecture: {model_name}")

    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(input_size, hidden_units)),
        ('relu1', nn.ReLU()),
        ('drop1', nn.Dropout(0.2)),
        ('fc2', nn.Linear(hidden_units, 512)),
        ('relu2', nn.ReLU()),
        ('drop2', nn.Dropout(0.2)),
        ('fc3', nn.Linear(512, num_classes)),
        ('output', nn.LogSoftmax(dim=1))
    ]))

    if model_name.startswith('vgg'):
        model.classifier = classifier
        optimizer = optim.Adam(model.classifier.parameters(), lr=lr)
    else:  # resnet
        model.fc = classifier
        optimizer = optim.Adam(model.fc.parameters(), lr=lr)

    criterion = nn.NLLLoss()
    return model, criterion, optimizer

def save_checkpoint(model, optimizer, train_data, filepath='checkpoint.pth', epochs=5, learning_rate=0.001, structure='vgg16'):
    model.class_to_idx = train_data.class_to_idx
    
    checkpoint = {
        'input_size': 25088,
        'output_size': len(train_data.classes),
        'structure': structure,
        'learning_rate': learning_rate,
        'classifier': model.classifier if hasattr(model, 'classifier') else model.fc,
        'epochs': epochs,
        'optimizer': optimizer.state_dict(),
        'state_dict': model.state_dict(),
        'class_to_idx': model.class_to_idx
    }
    
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    
    # Recreate the model architecture
    model, _, _ = nn_setup(
        model_name=checkpoint['structure'], 
        lr=checkpoint['learning_rate'],
        hidden_units=checkpoint['classifier'][0].out_features,
        num_classes=checkpoint['output_size']
    )
    
    # Load state dict
    model.load_state_dict(checkpoint['state_dict'])
    
    # Add class to index mapping
    model.class_to_idx = checkpoint['class_to_idx']
    
    return model

def validate(model, validloader, criterion, device='cuda'):
    model.eval()
    valid_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in validloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            batch_loss = criterion(log_ps, labels)
            valid_loss += batch_loss.item()
            ps = torch.exp(log_ps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    avg_valid_loss = valid_loss / len(validloader)
    avg_accuracy = accuracy / len(validloader)
    return avg_valid_loss, avg_accuracy

def train(model, trainloader, validloader, criterion, optimizer, epochs=3, print_every=5, device='cuda', early_stopping=None):
    steps = 0
    for e in range(epochs):
        running_loss = 0.0
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model.forward(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            
            if steps % print_every == 0:
                valid_loss, accuracy = validate(model, validloader, criterion, device)
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Loss: {running_loss/print_every:.3f}.. "
                      f"Validation Loss: {valid_loss:.3f}.. "
                      f"Accuracy: {accuracy:.3f}")
                running_loss = 0.0
                
                if early_stopping is not None and early_stopping(valid_loss):
                    print("Early stopping triggered.")
                    return

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0):
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss = float('inf')
        self.counter = 0

    def __call__(self, val_loss):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
        
        if self.counter >= self.patience:
            return True
        return False

def test_model(model, testloader, criterion, device='cuda'):
    model.eval()
    total_loss = 0.0
    total_accuracy = 0.0
    with torch.no_grad():
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            total_loss = loss.item() + total_loss 
            ps = torch.exp(log_ps)
            _, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            total_accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    
    avg_loss = total_loss / len(testloader)
    avg_accuracy = total_accuracy / len(testloader)
    
    print(f"Test Loss: {avg_loss:.3f}")
    print(f"Test Accuracy: {avg_accuracy:.3f}")
    
    return avg_loss, avg_accuracy
