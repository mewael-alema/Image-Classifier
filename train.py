import argparse
import os
import torch
import json

from model_utils import setup_model, save_checkpoint
from data_utils import prepare_dataloaders

def train(data_dir, save_dir, arch, learning_rate, hidden_units, epochs, gpu):
    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Prepare data loaders
    trainloader, validloader, _, train_data = prepare_dataloaders(data_dir)

    # Setup model
    model, criterion, optimizer = setup_model(
        arch=arch, 
        hidden_units=hidden_units, 
        num_classes=len(train_data.classes)
    )
    model.to(device)

    # Training loop
    steps = 0
    running_loss = 0
    print_every = 5

    for e in range(epochs):
        model.train()
        for inputs, labels in trainloader:
            steps += 1
            
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            
            log_ps = model.forward(inputs)
            loss = criterion(log_ps, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            
            # Validation pass
            if steps % print_every == 0:
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
                
                print(f"Epoch {e+1}/{epochs}.. "
                      f"Train loss: {running_loss/print_every:.3f}.. "
                      f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                      f"Validation accuracy: {accuracy/len(validloader):.3f}")
                
                running_loss = 0
                model.train()

    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)

    # Save checkpoint
    save_checkpoint(model, optimizer, train_data, save_dir, 
                    arch=arch, epochs=epochs, learning_rate=learning_rate)

def main():
    parser = argparse.ArgumentParser(description='Train a neural network')
    parser.add_argument('data_dir', type=str, help='Directory of training data')
    parser.add_argument('--save_dir', type=str, default='checkpoints', 
                        help='Directory to save checkpoints')
    parser.add_argument('--arch', type=str, default='vgg16', 
                        choices=['vgg16', 'vgg13', 'resnet50'], 
                        help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, 
                        help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=4096, 
                        help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=3, 
                        help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU for training')

    args = parser.parse_args()

    train(
        data_dir=args.data_dir, 
        save_dir=args.save_dir, 
        arch=args.arch, 
        learning_rate=args.learning_rate, 
        hidden_units=args.hidden_units, 
        epochs=args.epochs, 
        gpu=args.gpu
    )

if __name__ == '__main__':
    main()
