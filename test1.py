import torch
import torch.nn as nn
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import random
import os

def verify_predictions(model_path, data_dir='tomato/val', num_samples=5):
    """
    Verify model predictions by displaying images with both true and predicted labels.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the model
    model = DiseaseClassifier(num_classes=10).to(device)
    try:
        model.load_state_dict(torch.load(model_path, map_location=device))
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        return
    
    # Set model to evaluation mode
    model.eval()
    
    # Define transforms
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Load validation dataset
    try:
        dataset = datasets.ImageFolder(data_dir, transform=transform)
        print("\nClass mapping:")
        for class_name, idx in dataset.class_to_idx.items():
            print(f"{class_name}: {idx}")
    except Exception as e:
        print(f"Error loading dataset: {str(e)}")
        return
    
    # Create figure for visualization
    plt.figure(figsize=(20, 4))
    
    # Get random samples
    indices = random.sample(range(len(dataset)), num_samples)
    
    with torch.no_grad():
        for plot_idx, img_idx in enumerate(indices):
            # Get image and label
            image, true_label = dataset[img_idx]
            
            # Get prediction
            model_input = image.unsqueeze(0).to(device)
            output = model(model_input)
            _, predicted = output.max(1)
            
            # Convert image for display
            img_display = image.cpu().permute(1, 2, 0)
            img_display = img_display * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])
            img_display = img_display.numpy()
            img_display = np.clip(img_display, 0, 1)
            
            # Plot
            plt.subplot(1, num_samples, plot_idx + 1)
            plt.imshow(img_display)
            plt.title(f'True: {dataset.classes[true_label]}\nPred: {dataset.classes[predicted.item()]}', 
                     color='green' if true_label == predicted.item() else 'red')
            plt.axis('off')
    
    plt.tight_layout()
    plt.savefig('prediction_verification.png')
    plt.close()
    
    # Print model summary
    print("\nModel Architecture:")
    print(model)
    
    # Verify softmax outputs
    sample_image, _ = dataset[0]
    sample_output = model(sample_image.unsqueeze(0).to(device))
    softmax_output = nn.functional.softmax(sample_output, dim=1)
    
    print("\nSample prediction probabilities:")
    for class_name, prob in zip(dataset.classes, softmax_output[0].cpu().numpy()):
        print(f"{class_name}: {prob:.4f}")

if __name__ == "__main__":
    verify_predictions('best_model.pth')