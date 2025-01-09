import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import os
import matplotlib.pyplot as plt

def verify_data_loading():
    """
    Verify the data loading pipeline and display sample images with their labels.
    """
    # Basic transform for visualization
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    
    # Load training dataset
    try:
        dataset = datasets.ImageFolder('tomato/train', transform=transform)
        
        # Print folder structure and class mapping
        print("\nClass to Index Mapping:")
        for class_name, idx in dataset.class_to_idx.items():
            print(f"{class_name}: {idx}")
            
        print("\nFolder Structure:")
        for root, dirs, files in os.walk('tomato/train'):
            level = root.replace('tomato/train', '').count(os.sep)
            indent = ' ' * 4 * level
            print(f"{indent}{os.path.basename(root)}/")
        
        # Create a data loader with batch size 4 for visualization
        loader = DataLoader(dataset, batch_size=4, shuffle=True)
        
        # Get a batch of images
        images, labels = next(iter(loader))
        
        # Plot images with their labels
        plt.figure(figsize=(15, 5))
        for idx in range(4):
            plt.subplot(1, 4, idx + 1)
            plt.imshow(images[idx].permute(1, 2, 0))
            plt.title(f"Label: {dataset.classes[labels[idx]]}")
            plt.axis('off')
        plt.savefig('sample_data_verification.png')
        plt.close()
        
        # Print dataset statistics
        print(f"\nTotal number of images: {len(dataset)}")
        print("\nSamples per class:")
        class_counts = [0] * len(dataset.classes)
        for _, label in dataset:
            class_counts[label] += 1
        for class_name, count in zip(dataset.classes, class_counts):
            print(f"{class_name}: {count} images")
            
    except Exception as e:
        print(f"Error during verification: {str(e)}")

if __name__ == "__main__":
    verify_data_loading()