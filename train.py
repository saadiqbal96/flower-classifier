# train.py
import argparse
import json
import time
import copy
import torch
import torch.nn as nn
from torch import optim
from torchvision import datasets, transforms, models

def get_input_args():
    parser = argparse.ArgumentParser(description='Train a deep learning model for image classification')
    parser.add_argument('data_dir', help='Path to dataset directory (should include train/ valid/ test subfolders)')
    parser.add_argument('--save_dir', default='saved_checkpoints', help='Directory to save checkpoints')
    parser.add_argument('--arch', default='vgg16', choices=['vgg16', 'densenet121'], help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=512, help='Number of units in hidden layer')
    parser.add_argument('--epochs', type=int, default=5, help='Number of training epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def build_model(arch='vgg16', hidden_units=512, output_size=102):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features  # 25088
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features    # 1024
    else:
        raise ValueError("Unsupported architecture")
    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    classifier = nn.Sequential(
        nn.Linear(input_size, hidden_units),
        nn.ReLU(),
        nn.Dropout(0.2),
        nn.Linear(hidden_units, output_size),
        nn.LogSoftmax(dim=1)
    )
    # Attach classifier
    if arch == 'vgg16':
        model.classifier = classifier
    else:
        model.classifier = classifier
    return model

def data_loaders(data_dir):
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir  = data_dir + '/test'

    train_transforms = transforms.Compose([
        transforms.RandomRotation(30),
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])
    test_valid_transforms = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
    ])

    train_dataset = datasets.ImageFolder(train_dir, transform=train_transforms)
    valid_dataset = datasets.ImageFolder(valid_dir, transform=test_valid_transforms)
    test_dataset  = datasets.ImageFolder(test_dir,  transform=test_valid_transforms)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=64)
    test_loader  = torch.utils.data.DataLoader(test_dataset, batch_size=64)

    return train_loader, valid_loader, test_loader, train_dataset.class_to_idx

def validate(model, valid_loader, criterion, device):
    model.eval()
    val_loss = 0
    accuracy = 0
    with torch.no_grad():
        for inputs, labels in valid_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model.forward(inputs)
            batch_loss = criterion(outputs, labels)
            val_loss += batch_loss.item()

            ps = torch.exp(outputs)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class.view(labels.shape) == labels
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
    return val_loss/len(valid_loader), accuracy/len(valid_loader)

def train():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    train_loader, valid_loader, test_loader, class_to_idx = data_loaders(args.data_dir)
    model = build_model(arch=args.arch, hidden_units=args.hidden_units)
    model.to(device)

    criterion = nn.NLLLoss()
    # Only train parameters of classifier
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate)

    steps = 0
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    start_time = time.time()
    for epoch in range(args.epochs):
        model.train()
        running_loss = 0
        for inputs, labels in train_loader:
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # Validation
        val_loss, val_accuracy = validate(model, valid_loader, criterion, device)
        print(f"Epoch {epoch+1}/{args.epochs}.. "
              f"Train loss: {running_loss/len(train_loader):.3f}.. "
              f"Validation loss: {val_loss:.3f}.. "
              f"Validation accuracy: {val_accuracy:.3f}")

        # save best
        if val_accuracy > best_acc:
            best_acc = val_accuracy
            best_model_wts = copy.deepcopy(model.state_dict())

    total_time = time.time() - start_time
    print(f"Training complete in {total_time//60:.0f}m {total_time%60:.0f}s. Best val acc: {best_acc:.3f}")

    # Load best weights
    model.load_state_dict(best_model_wts)

    # Test accuracy
    test_loss, test_accuracy = validate(model, test_loader, criterion, device)
    print(f"Test accuracy: {test_accuracy:.3f}")

    # Save checkpoint
    import os
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    save_path = os.path.join(args.save_dir, 'checkpoint.pth')
    checkpoint = {
        'arch': args.arch,
        'class_to_idx': class_to_idx,
        'state_dict': model.state_dict(),
        'hidden_units': args.hidden_units,
        'output_size': 102,
        'learning_rate': args.learning_rate,
        'epochs': args.epochs
    }
    torch.save(checkpoint, save_path)
    print(f"Checkpoint saved to {save_path}")

if __name__ == '__main__':
    train()
