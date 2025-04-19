import argparse
import json
import torch
import numpy as np
from PIL import Image

from model_utils import load_checkpoint
from data_utils import process_image

def predict(image_path, checkpoint, topk, category_names, gpu):
    # Set device
    device = torch.device("cuda" if gpu and torch.cuda.is_available() else "cpu")
    
    # Load the checkpoint
    model = load_checkpoint(checkpoint)
    model.to(device)
    model.eval()

    # Process the image
    image = process_image(image_path)
    image = image.unsqueeze(0).to(device)

    # Get the probabilities
    with torch.no_grad():
        log_ps = model(image)
        ps = torch.exp(log_ps)
        top_ps, top_classes = ps.topk(topk, dim=1)

    # Convert to numpy
    top_ps = top_ps.cpu().numpy()[0]
    top_classes = top_classes.cpu().numpy()[0]

    # Convert indices to classes
    idx_to_class = {val: key for key, val in model.class_to_idx.items()}
    top_classes = [idx_to_class[cls] for cls in top_classes]

    # Load category names if provided
    if category_names:
        with open(category_names, 'r') as f:
            cat_to_name = json.load(f)
        top_classes = [cat_to_name.get(cls, cls) for cls in top_classes]

    return top_ps, top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the input image')
    parser.add_argument('checkpoint', type=str, help='Path to checkpoint file')
    parser.add_argument('--top_k', type=int, default=1, 
                        help='Return top K most likely classes')
    parser.add_argument('--category_names', type=str, 
                        help='JSON file mapping categories to real names')
    parser.add_argument('--gpu', action='store_true', 
                        help='Use GPU for inference')

    args = parser.parse_args()

    # Predict
    probs, classes = predict(
        image_path=args.image_path, 
        checkpoint=args.checkpoint, 
        topk=args.top_k, 
        category_names=args.category_names, 
        gpu=args.gpu
    )

    # Print results
    for prob, cls in zip(probs, classes):
        print(f"{cls}: {prob*100:.2f}%")

if __name__ == '__main__':
    main()
