import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from tqdm import tqdm
import os
import json

# Constants
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 50
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

class LabelMapper:
    def __init__(self, dataset_path):
        """Initialize label mapper and save mapping to file."""
        self.train_dir = os.path.join(dataset_path, 'train')
        self.class_names = sorted(os.listdir(self.train_dir))  # Sort for consistency
        self.num_classes = len(self.class_names)
        
        # Create and save the mapping
        self.label_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
        self.idx_to_label = {idx: name for name, idx in self.label_to_idx.items()}
        
        # Save mapping to file for verification
        mapping_file = 'label_mapping.json'
        with open(mapping_file, 'w') as f:
            json.dump({
                'label_to_idx': self.label_to_idx,
                'idx_to_label': self.idx_to_label
            }, f, indent=4)
        
        print("Label mapping saved to", mapping_file)
        print("\nClass mapping:")
        for name, idx in self.label_to_idx.items():
            print(f"{idx}: {name}")

    def verify_dataset_labels(self, dataset):
        """Verify that dataset labels match our mapping."""
        if not hasattr(dataset, 'classes'):
            return False
        
        dataset_classes = dataset.classes
        if len(dataset_classes) != len(self.class_names):
            print("Warning: Number of classes mismatch!")
            return False
            
        for i, class_name in enumerate(dataset_classes):
            if class_name != self.class_names[i]:
                print(f"Warning: Label mismatch at index {i}")
                print(f"Expected: {self.class_names[i]}, Got: {class_name}")
                return False
        return True

class DiseaseClassifier(nn.Module):
    def __init__(self, num_classes):
        super(DiseaseClassifier, self).__init__()
        self.model = models.mobilenet_v2(pretrained=True)
        
        # Freeze the pre-trained layers
        for param in self.model.parameters():
            param.requires_grad = False
            
        self.model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(self.model.last_channel, 512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.model(x)

def create_data_loaders(dataset_path, label_mapper):
    """Create training and validation data loaders with label verification."""
    train_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                           std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets with explicit class mapping
    train_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'train'),
        transform=train_transform,
        target_transform=None  # We'll verify rather than transform
    )
    
    val_dataset = datasets.ImageFolder(
        os.path.join(dataset_path, 'val'),
        transform=val_transform,
        target_transform=None
    )
    
    # Verify label alignment
    print("\nVerifying training dataset labels...")
    if not label_mapper.verify_dataset_labels(train_dataset):
        raise ValueError("Training dataset labels don't match the expected mapping!")
        
    print("\nVerifying validation dataset labels...")
    if not label_mapper.verify_dataset_labels(val_dataset):
        raise ValueError("Validation dataset labels don't match the expected mapping!")
    
    # Create data loaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, 
                            shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, 
                          shuffle=False, num_workers=4)
    
    return train_loader, val_loader

def verify_batch_labels(labels, predictions, label_mapper):
    """Verify that the labels in a batch are consistent with our mapping."""
    for label in labels:
        if label >= len(label_mapper.class_names):
            print(f"Warning: Invalid label index {label}")
            return False
    return True

def train_model(model, train_loader, val_loader, criterion, optimizer, label_mapper, num_epochs):
    """Train the model with label verification at each step."""
    best_val_acc = 0.0
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}'):
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            
            # Verify labels before training
            if not verify_batch_labels(labels, None, label_mapper):
                raise ValueError("Label verification failed during training!")
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100. * correct / total
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                
                # Verify labels during validation
                if not verify_batch_labels(labels, None, label_mapper):
                    raise ValueError("Label verification failed during validation!")
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()
        
        val_loss = val_loss / len(val_loader)
        val_acc = 100. * correct / total
        
        print(f'\nEpoch {epoch+1}/{num_epochs}:')
        print(f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%')
        print(f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%')
        
        # Save best model with label mapping
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'model_state_dict': model.state_dict(),
                'label_mapping': label_mapper.label_to_idx,
                'val_acc': val_acc
            }, 'best_model.pth')

def evaluate_model(model, val_loader, label_mapper):
    """Evaluate model performance with explicit label mapping."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(DEVICE)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            
            # Convert to actual class names for reporting
            pred_classes = [label_mapper.idx_to_label[idx.item()] 
                          for idx in predicted]
            true_classes = [label_mapper.idx_to_label[idx.item()] 
                          for idx in labels]
            
            all_preds.extend(pred_classes)
            all_labels.extend(true_classes)
    
    # Generate classification report with actual class names
    report = classification_report(all_labels, all_preds)
    print("\nClassification Report:\n", report)
    
    # Generate confusion matrix with actual class names
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(12, 8))
    sns.heatmap(cm, annot=True, fmt='d', 
                xticklabels=label_mapper.class_names,
                yticklabels=label_mapper.class_names)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('confusion_matrix.png')

if __name__ == "__main__":
    # Initialize label mapper
    dataset_path = 'tomato'  # Your dataset path
    label_mapper = LabelMapper(dataset_path)
    
    # Create model and move to device
    model = DiseaseClassifier(label_mapper.num_classes).to(DEVICE)
    
    # Create data loaders with label verification
    train_loader, val_loader = create_data_loaders(dataset_path, label_mapper)
    
    # Define loss function and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    
    # Train the model with label verification
    train_model(model, train_loader, val_loader, criterion, optimizer, 
                label_mapper, EPOCHS)
    
    # Evaluate the model with explicit label mapping
    evaluate_model(model, val_loader, label_mapper)