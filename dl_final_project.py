# Import necessary libraries for data processing, modeling, and evaluation
from torchvision import transforms
from transformers import DistilBertTokenizer, BertModel, AdamW
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score, f1_score, classification_report
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from torchvision import models
from torchvision.transforms.functional import pad
from sklearn.utils.class_weight import compute_class_weight

# ===========================
# Define the ImageModel Class
# ===========================
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        # Load a pre-trained ResNet-18 model
        self.resnet = models.resnet18(pretrained=True)

        # Freeze all layers in ResNet to prevent them from being updated during training
        for param in self.resnet.parameters():
            param.requires_grad = False

        # Unfreeze the last two residual layers (layer3 and layer4) for fine-tuning
        for param in self.resnet.layer4.parameters():
            param.requires_grad = True
        for param in self.resnet.layer3.parameters():
            param.requires_grad = True

        # Modify the fully connected layer to output 128-dimensional features
        self.resnet.fc = nn.Sequential(
            nn.Dropout(0.6),          # Add dropout for regularization
            nn.Linear(512, 128),      # Change input features from 512 to 128
            nn.ReLU()                  # Apply ReLU activation
        )

    def forward(self, x):
        # Forward pass through ResNet
        return self.resnet(x)

# ==========================
# Define the TextModel Class
# ==========================
class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        # Load a pre-trained BERT model
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Freeze all BERT parameters
        for param in self.bert.parameters():
            param.requires_grad = False

        # Unfreeze the last two encoder layers for fine-tuning
        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        # Add a fully connected layer to project BERT's output to 128 dimensions
        self.fc = nn.Sequential(
            nn.Dropout(0.6),          # Add dropout for regularization
            nn.Linear(768, 128),      # Change input features from 768 to 128
            nn.ReLU()                  # Apply ReLU activation
        )

    def forward(self, input_ids, attention_mask):
        # Forward pass through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Extract the pooled output ([CLS] token)
        pooled_output = outputs.pooler_output
        # Pass through the fully connected layer
        return self.fc(pooled_output)

# ============================
# Define the MultimodalModel Class
# ============================
class MultimodalModel(nn.Module):
    def __init__(self, image_dim=128, text_dim=128, num_heads=4, fusion_dim=256):
        super(MultimodalModel, self).__init__()
        # Initialize image and text models
        self.image_model = ImageModel()
        self.text_model = TextModel()

        # Define a linear layer to compute attention weights for image and text features
        self.attention_fc = nn.Linear(256, 2)  # Outputs attention weights for image and text
        self.softmax = nn.Softmax(dim=1)       # Softmax to normalize attention weights

        # Define projection layers to align image and text feature dimensions for fusion
        self.image_projection = nn.Linear(image_dim, fusion_dim)
        self.text_projection = nn.Linear(text_dim, fusion_dim)

        # Multi-head attention mechanism for feature fusion
        self.multihead_attention = nn.MultiheadAttention(
            embed_dim=fusion_dim,
            num_heads=num_heads,
            dropout=0.6,
            batch_first=True
        )

        # Additional dropout for regularization
        self.attention_dropout = nn.Dropout(0.6)

        # Define the classifier that maps fused features to output classes
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),    # First linear layer
            nn.ReLU(),              # ReLU activation
            nn.Dropout(0.6),        # Dropout for regularization
            nn.Linear(128, 7)       # Output layer with 7 classes
        )

        # General dropout layer
        self.dropout = nn.Dropout(0.6)

    def forward(self, input_ids, attention_mask, images):
        # Extract image features using the image model
        image_features = self.image_model(images)
        image_features = self.dropout(image_features)

        # Extract text features using the text model
        text_features = self.text_model(input_ids, attention_mask)
        text_features = self.dropout(text_features)

        # Project image and text features to the fusion dimension
        image_features_proj = self.image_projection(image_features).unsqueeze(1)  # Shape: (batch_size, 1, fusion_dim)
        text_features_proj = self.text_projection(text_features).unsqueeze(1)    # Shape: (batch_size, 1, fusion_dim)

        # Concatenate projected features along the sequence dimension
        combined_features = torch.cat((image_features_proj, text_features_proj), dim=1)  # Shape: (batch_size, 2, fusion_dim)
        
        # Apply multi-head attention to the combined features
        attn_output, _ = self.multihead_attention(combined_features, combined_features, combined_features)
        
        # Aggregate the attended features by taking the mean across the sequence length
        fused_features = attn_output.mean(dim=1)  # Shape: (batch_size, fusion_dim)
        
        # Pass the fused features through the classifier to get output logits
        output = self.classifier(fused_features)

        return output

# ===================================
# Define the MultimodalDataset Class
# ===================================
class MultimodalDataset(Dataset):
    def __init__(self, data, transform, tokenizer, folder_path):
        """
        Args:
            data (DataFrame): Pandas DataFrame containing the dataset.
            transform (transforms.Compose): Transformations to apply to images.
            tokenizer (DistilBertTokenizer): Tokenizer for processing text.
            folder_path (str): Path to the folder containing image files.
        """
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.folder_path = folder_path
        # Define a mapping from emotion labels to numerical indices
        self.labels = {
            'neutral': 0, 'joy': 1, 'sadness': 2,
            'fear': 3, 'anger': 4, 'surprise': 5, 'disgust': 6
        }

    def __len__(self):
        # Return the total number of samples
        return len(self.data)

    def __getitem__(self, idx):
        # Get the row corresponding to the index
        row = self.data.iloc[idx]
        
        # Extract text and label
        text = row['Utterance']
        label = self.labels[row['Emotion']]

        # Handle cases where text might not be a string
        if type(text) != str:
            text = ""

        # Construct the image file path based on Dialogue_ID and Utterance_ID
        image_path = f"{self.folder_path}/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.jpg"
        # Open and convert the image to RGB
        image = Image.open(image_path).convert('RGB')

        # Apply image transformations
        image = self.transform(image)
    
        # Process text to obtain input_ids and attention_mask
        input_ids, attention_mask = preprocess_text(text, tokenizer=self.tokenizer)

        # Convert label to a tensor
        label = torch.tensor(label, dtype=torch.long)

        # Return a dictionary of processed features and label
        return {
            'image': image,
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'label': label
        }

# ================================
# Define the SquarePad Transformation
# ================================
class SquarePad:
    def __call__(self, image):
        """
        Pads the image to make it square by adding borders.
        Args:
            image (PIL.Image): Image to be padded.
        Returns:
            PIL.Image: Padded square image.
        """
        width, height = image.size
        max_dim = max(width, height)
        # Calculate padding sizes
        left = (max_dim - width) // 2
        right = max_dim - width - left
        top = (max_dim - height) // 2
        bottom = max_dim - height - top
        # Apply padding
        return pad(image, (left, top, right, bottom), fill=0, padding_mode='constant')

# ===============================
# Function to Compute Class Weights
# ===============================
def compute_weights(data):
    """
    Computes class weights to handle class imbalance.
    Args:
        data (DataFrame): DataFrame containing the dataset.
    Returns:
        torch.Tensor: Tensor of class weights.
    """
    # Map emotion labels to numerical indices
    labels = data['Emotion'].map({
        'neutral': 0, 'joy': 1, 'sadness': 2,
        'fear': 3, 'anger': 4, 'surprise': 5, 'disgust': 6
    }).values
    # Compute class weights using sklearn
    class_weights = compute_class_weight('balanced', classes=np.unique(labels), y=labels)
    # Convert to torch tensor
    return torch.tensor(class_weights, dtype=torch.float)

# ================================
# Function to Load and Prepare Data
# ================================
def get_data():
    """
    Loads the training and validation data, applies transformations,
    and creates DataLoaders.
    Returns:
        DataLoader: Training DataLoader.
        DataLoader: Validation DataLoader.
        nn.CrossEntropyLoss: Loss criterion with class weights.
    """
    # Load training and validation CSV files
    train_data = pd.read_csv('Archive/combinedTrainData/trainMELD.csv', encoding='latin1')
    validation_data = pd.read_csv('Archive/combinedValData/valMELD.csv', encoding='latin1')

    # Print the shapes of the datasets
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")

    # Determine the device to use (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Compute class weights and move them to the device
    class_weights = compute_weights(train_data).to(device)
    # Define the loss criterion with class weights to handle class imbalance
    criterion = CrossEntropyLoss(weight=class_weights)

    # ======================
    # Define Image Transformations
    # ======================
    transform = transforms.Compose([
        SquarePad(),                                    # Make the image square
        transforms.Resize((128, 128)),                  # Resize to 128x128 pixels

        # Data augmentation techniques for better generalization
        transforms.RandomHorizontalFlip(p=0.5),         # Random horizontal flip with 50% probability
        transforms.RandomRotation(degrees=15),           # Random rotation up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Random color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)), # Random crop and resize

        transforms.ToTensor(),                           # Convert PIL image to tensor
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # Normalize the image
    ])

    # Initialize the tokenizer for text processing
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    # Create dataset instances for training and validation
    train_dataset = MultimodalDataset(train_data, transform, tokenizer, 'Archive/combinedTrainData/processedFrames')
    validation_dataset = MultimodalDataset(validation_data, transform, tokenizer, 'Archive/combinedValData/processedFrames')

    # Print the sizes of the datasets
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    # Create DataLoaders for batching and shuffling
    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    return train_dataloader, validation_dataloader, criterion

# ================================
# Function to Preprocess Text
# ================================
def preprocess_text(text, tokenizer):
    """
    Tokenizes and encodes the input text using the provided tokenizer.
    Args:
        text (str): The input text to preprocess.
        tokenizer (DistilBertTokenizer): The tokenizer to use.
    Returns:
        torch.Tensor: Input IDs tensor.
        torch.Tensor: Attention mask tensor.
    """
    # Tokenize and encode the text, padding/truncating to a max length of 50
    encoded = tokenizer(
        text,
        padding='max_length',
        max_length=50,
        truncation=True,
        return_tensors="pt"  # Return PyTorch tensors
    )
    # Squeeze to remove the batch dimension
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

# =========================================
# Function to Train the Multimodal Model
# =========================================
def train_model(train_dataloader, validation_dataloader, criterion):
    """
    Trains the multimodal model using the provided DataLoaders and criterion.
    Args:
        train_dataloader (DataLoader): DataLoader for training data.
        validation_dataloader (DataLoader): DataLoader for validation data.
        criterion (nn.CrossEntropyLoss): Loss function with class weights.
    """
    # Determine the device to use (GPU if available)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # Initialize the multimodal model and move it to the device
    model = MultimodalModel().to(device)

    # Define the optimizer with different learning rates for different parts of the model
    optimizer = AdamW([
        # ResNet layers with lower learning rate
        {'params': model.image_model.resnet.layer3.parameters(), 'lr': 1e-5},
        {'params': model.image_model.resnet.layer4.parameters(), 'lr': 1e-5},
        {'params': model.image_model.resnet.fc.parameters(), 'lr': 1e-4},

        # BERT layers with lower learning rate
        {'params': model.text_model.bert.encoder.layer[-4:].parameters(), 'lr': 1e-5},
        {'params': model.text_model.fc.parameters(), 'lr': 1e-4},

        # Fusion classifier with higher learning rate
        {'params': model.classifier.parameters(), 'lr': 1e-4}
    ], weight_decay=1e-3)  # Weight decay for regularization
    
    # Lists to store training and validation loss per epoch
    train_loss = []
    val_loss = []
    # Parameters for early stopping
    patience = 5
    count = 0
    best_model = None
    best_loss = float('inf')

    # Set the model to training mode
    model.train()
    # Loop over epochs
    for epoch in range(30):
        total_loss = 0
        # Loop over batches in the training DataLoader
        for i, batch in enumerate(train_dataloader):
            print(f"Batch {i}")  # Debug: Print current batch number
            # Move data to the device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            # Zero the gradients
            optimizer.zero_grad()
            # Forward pass through the model
            outputs = model(input_ids, attention_mask, image)
            # Compute the loss
            loss = criterion(outputs, labels)
            # Backward pass to compute gradients
            loss.backward()
            # Update the model parameters
            optimizer.step()
            
            # Accumulate the loss
            total_loss += loss.item()
        
        # Calculate average training loss for the epoch
        avg_train_loss = total_loss / len(train_dataloader)
        train_loss.append(avg_train_loss)

        print(f"Epoch {epoch+1}, Loss: {avg_train_loss}")

        # Evaluate the model on the validation set
        accuracy, avg_val_loss = evaluate_model(validation_dataloader, model, device, criterion)
        val_loss.append(avg_val_loss)

        # Check if the current validation loss is the best so far
        if avg_val_loss < best_loss:
            best_loss = avg_val_loss
            best_model = model
            count = 0  # Reset patience counter
        else:
            count += 1  # Increment patience counter

        # Early stopping: If no improvement for 'patience' epochs, stop training
        if count >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    # Plot the training and validation loss curves
    plot_results(train_loss, val_loss)
    # Save the best model's state_dict
    torch.save(best_model.state_dict(), 'best_model_2.pth')

# =====================================
# Function to Plot Training and Validation Loss
# =====================================
def plot_results(train_loss, val_loss):
    """
    Plots the training and validation loss over epochs.
    Args:
        train_loss (list): List of training losses per epoch.
        val_loss (list): List of validation losses per epoch.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss Over Epochs')
    plt.show() 

# =====================================
# Function to Evaluate the Model
# =====================================
def evaluate_model(dataloader, model, device, criterion):
    """
    Evaluates the model on the validation dataset.
    Args:
        dataloader (DataLoader): DataLoader for validation data.
        model (nn.Module): The trained model to evaluate.
        device (torch.device): Device to perform computations on.
        criterion (nn.CrossEntropyLoss): Loss function with class weights.
    Returns:
        float: Accuracy of the model on the validation set.
        float: Average loss on the validation set.
    """
    # Set the model to evaluation mode
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0

    # Lists to store predictions and true labels for metrics
    predictions = []
    true_labels = []

    # Disable gradient computation for evaluation
    with torch.no_grad():
        for batch in dataloader:
            # Move data to the device
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            # Forward pass through the model
            outputs = model(input_ids, attention_mask, image)
            # Compute the loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get the predicted class by taking the argmax
            _, predicted = torch.max(outputs, 1)
            # Update the count of correct predictions
            correct_predictions += (predicted == labels).sum().item()
            # Update the total number of samples
            total_samples += labels.size(0)

            # Collect predictions and true labels for metrics
            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    # Calculate accuracy
    accuracy = correct_predictions / total_samples
    # Compute the confusion matrix
    confusion = confusion_matrix(true_labels, predictions)
    print(f"Confusion Matrix:\n{confusion}")
    # Compute F1 scores for each class
    f1_scores = f1_score(true_labels, predictions, average=None)
    print(f"F1 Scores: {f1_scores}")
    # Compute precision scores for each class
    precision_scores = precision_score(true_labels, predictions, average=None)
    print(f"Precision Scores: {precision_scores}")
    # Compute recall scores for each class
    recall_scores = recall_score(true_labels, predictions, average=None)
    print(f"Recall Scores: {recall_scores}")
    # Generate a full classification report
    classification_report_str = classification_report(true_labels, predictions)
    print(f"Classification Report:\n{classification_report_str}")
    # Calculate average validation loss
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss}, Accuracy: {accuracy}")

    return accuracy, avg_loss

# ===============================
# Main Function to Execute Training
# ===============================
def main():
    """
    Main function to load data, train the model, and evaluate its performance.
    """
    # Load DataLoaders and criterion
    train_dataloader, validation_dataloader, criterion = get_data()
    # Start training the model
    train_model(train_dataloader, validation_dataloader, criterion)

# ===============================
# Entry Point of the Script
# ===============================
if __name__ == '__main__':
    main()
