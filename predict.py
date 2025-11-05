# predict.py
import argparse
import json
import torch
from torchvision import models
from PIL import Image
import numpy as np
import torch.nn.functional as F

def get_input_args():
    parser = argparse.ArgumentParser(description='Predict image class using a trained model')
    parser.add_argument('image_path', help='Path to image')
    parser.add_argument('checkpoint', help='Path to checkpoint .pth file')
    parser.add_argument('--top_k', type=int, default=5, help='Return top K predictions')
    parser.add_argument('--category_names', help='JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU if available')
    return parser.parse_args()

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath, map_location='cpu')
    arch = checkpoint.get('arch', 'vgg16')
    hidden_units = checkpoint.get('hidden_units', 512)
    output_size = checkpoint.get('output_size', 102)

    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        input_size = model.classifier[0].in_features
    elif arch == 'densenet121':
        model = models.densenet121(pretrained=True)
        input_size = model.classifier.in_features
    else:
        raise ValueError("Unsupported architecture in checkpoint")

    # freeze
    for param in model.parameters():
        param.requires_grad = False

    # rebuild classifier (must match the one saved)
    classifier = torch.nn.Sequential(
        torch.nn.Linear(input_size, hidden_units),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden_units, output_size),
        torch.nn.LogSoftmax(dim=1)
    )
    model.classifier = classifier
    model.load_state_dict(checkpoint['state_dict'])
    model.class_to_idx = checkpoint['class_to_idx']
    return model

def process_image(image_path):
    pil_image = Image.open(image_path).convert("RGB")

    # Resize such that shorter side = 256 keeping aspect ratio
    width, height = pil_image.size
    if width < height:
        new_width = 256
        new_height = int(256 * height / width)
    else:
        new_height = 256
        new_width = int(256 * width / height)
    pil_image = pil_image.resize((new_width, new_height))

    # Center crop 224x224
    left = (pil_image.width - 224) / 2
    top = (pil_image.height - 224) / 2
    right = left + 224
    bottom = top + 224
    pil_image = pil_image.crop((left, top, right, bottom))

    # Convert to numpy, scale, and normalize
    np_image = np.array(pil_image) / 255.0
    mean = np.array([0.485,0.456,0.406])
    std  = np.array([0.229,0.224,0.225])
    np_image = (np_image - mean) / std

    # Reorder dimensions to match PyTorch (C,H,W)
    np_image = np_image.transpose((2,0,1))
    return torch.from_numpy(np_image).type(torch.FloatTensor)

def predict(image_path, model, topk=5, device='cpu'):
    model.to(device)
    model.eval()
    img = process_image(image_path).unsqueeze(0).to(device)  # add batch
    with torch.no_grad():
        output = model(img)
        ps = torch.exp(output)
        top_p, top_class = ps.topk(topk, dim=1)
    top_p = top_p.cpu().numpy().tolist()[0]
    top_class = top_class.cpu().numpy().tolist()[0]

    # Map indices to classes
    idx_to_class = {v:k for k,v in model.class_to_idx.items()}
    top_labels = [idx_to_class[c] for c in top_class]
    return top_p, top_labels

def main():
    args = get_input_args()
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model = load_checkpoint(args.checkpoint)
    top_p, top_labels = predict(args.image_path, model, topk=args.top_k, device=device)

    # Load category names if provided
    cat_to_name = None
    if args.category_names:
        with open(args.category_names, 'r') as f:
            cat_to_name = json.load(f)

    print("Top K predictions:")
    for i, (p, label) in enumerate(zip(top_p, top_labels), 1):
        name = cat_to_name[label] if cat_to_name and label in cat_to_name else label
        print(f"{i}: {name} (class {label}) with probability {p:.4f}")

if __name__ == '__main__':
    main()
