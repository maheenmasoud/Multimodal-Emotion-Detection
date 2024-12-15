from torchvision import transforms
from transformers import DistilBertTokenizer
import torch
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from transformers import DistilBertModel
import torch.nn as nn
from torch.optim import Adam
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, confusion_matrix
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from transformers import BertModel
from torchvision import models
from PIL import Image
from torchvision.transforms.functional import pad

# ResNet Model
class ImageModel(nn.Module):
    def __init__(self):
        super(ImageModel, self).__init__()
        self.resnet = models.resnet18(pretrained=True)

        # Change the last layer
        for param in self.resnet.parameters():
            param.requires_grad = False

        for param in self.resnet.layer4.parameters():
            param.requires_grad = True

        self.resnet.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(512, 128), nn.ReLU())

    def forward(self, x):
        return self.resnet(x)

# Text Model
class TextModel(nn.Module):
    def __init__(self):
        super(TextModel, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')

        # Change the last layer
        for param in self.bert.parameters():
            param.requires_grad = False

        for param in self.bert.encoder.layer[-2:].parameters():
            param.requires_grad = True

        self.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(768, 128), nn.ReLU())

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled_output = outputs.pooler_output
        return self.fc(pooled_output)
    
# Fusion Model
class MultimodalModel(nn.Module):
    def __init__(self):
        super(MultimodalModel, self).__init__()
        self.image_model = ImageModel()
        self.text_model = TextModel()
        self.classifier = nn.Linear(256, 7)

    def forward(self, input_ids, attention_mask, images):
        image_features = self.image_model(images)
        text_features = self.text_model(input_ids, attention_mask)
        combined_features = torch.cat((image_features, text_features), dim=1)
        return self.classifier(combined_features)

# class MultimodalEmotionModel(torch.nn.Module):
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#         # Image branch
#         self.image_branch = nn.Sequential(
#             nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
#             nn.ReLU(),
#             nn.MaxPool2d(2),
#             nn.Flatten(),
#             nn.Linear(32 * 32 * 32, 256),
#             nn.ReLU()
#         )

#         # Text branch
#         self.text_branch = DistilBertModel.from_pretrained('distilbert-base-uncased')
#         self.text_fc = nn.Linear(768, 256)

#         # Combined branch
#         self.fc = nn.Sequential(
#             nn.Linear(512, 128),
#             nn.ReLU(),
#             nn.Linear(128, 7)  # Assuming 7 emotion classes
#         )

#     def forward(self, image, input_ids, attention_mask):
#         # Image path
#         image_features = self.image_branch(image)

#         # Text path
#         text_features = self.text_branch(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
#         text_features = self.text_fc(text_features)

#         # Combine
#         combined = torch.cat((image_features, text_features), dim=1)
#         output = self.fc(combined)

#         return output


class MultimodalDataset(Dataset):
    def __init__(self, data, transform, tokenizer, folder_path):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform
        self.folder_path = folder_path
        self.labels = {'neutral': 0, 'joy': 1, 'sadness': 2, 'fear': 3, 'anger': 4, 'surprise': 5, 'disgust': 6}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text = row['Utterance']
        label = self.labels[row['Emotion']]
        # print(text, type(text))
        # print(label, type(label))
        # print(f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.jpg")
        if type(text) != str:
            text = ""

        # path for image file
        image_path = f"{self.folder_path}/dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}.jpg"
        image = Image.open(image_path).convert('RGB')
        #print(f"dia{row['Dialogue_ID']}_utt{row['Utterance_ID']}")

        # Process image
        image = self.transform(image)
    
        # Process text
        input_ids, attention_mask = preprocess_text(text, tokenizer=self.tokenizer)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
    

class SquarePad:
    def __call__(self, image):
        width, height = image.size
        max_dim = max(width, height)
        left = (max_dim - width) // 2
        right = max_dim - width - left
        top = (max_dim - height) // 2
        bottom = max_dim - height - top
        return pad(image, (left, top, right, bottom), fill=0, padding_mode='constant')


def get_data():
    train_data = pd.read_csv('Archive/combinedTrainData/trainMELD.csv', encoding='latin1')
    validation_data = pd.read_csv('Archive/combinedValData/valMELD.csv', encoding='latin1')
    print(f"Train data shape: {train_data.shape}")
    print(f"Validation data shape: {validation_data.shape}")

    # Image preprocessing
    transform = transforms.Compose([
        SquarePad(),
        transforms.Resize((128, 128)),

        # Data augmentation transformations
        transforms.RandomHorizontalFlip(p=0.5),  # Randomly flip image horizontally
        transforms.RandomRotation(degrees=15),   # Random rotation up to 15 degrees
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # Color jitter
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
        transforms.RandomResizedCrop(size=128, scale=(0.8, 1.0)),  # Random crop and resize


        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

    # Text preprocessing
    tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

    train_dataset = MultimodalDataset(train_data, transform, tokenizer, 'Archive/combinedTrainData/processedFrames')
    validation_dataset = MultimodalDataset(validation_data, transform, tokenizer, 'Archive/combinedValData/processedFrames')

    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(validation_dataset)}")

    train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    validation_dataloader = DataLoader(validation_dataset, batch_size=32, shuffle=False)

    return train_dataloader, validation_dataloader

def preprocess_text(text, tokenizer):
    encoded = tokenizer(text, padding='max_length', max_length=50, truncation=True, return_tensors="pt") # could change max_length based on the data
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

def train_model(train_dataloader, validation_dataloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalModel().to(device)

    optimizer = Adam([
    # ResNet layers
    {'params': model.image_model.resnet.layer4.parameters(), 'lr': 1e-5},
    {'params': model.image_model.resnet.fc.parameters(), 'lr': 1e-4},

    # BERT layers
    {'params': model.text_model.bert.encoder.layer[-2:].parameters(), 'lr': 1e-5},
    {'params': model.text_model.fc.parameters(), 'lr': 1e-4},

    # Fusion classifier
    {'params': model.classifier.parameters(), 'lr': 1e-4}
], weight_decay=1e-5)
    
    criterion = CrossEntropyLoss()

    train_loss = []
    val_loss = []

    model.train()
    for epoch in range(2):
        total_loss = 0
        for i, batch in enumerate(train_dataloader):
            print(f"Batch {i}")
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            outputs = model(input_ids, attention_mask, image)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        train_loss.append(total_loss / len(train_dataloader))

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(train_dataloader)}")

        accuracy, avg_loss = evaluate_model(validation_dataloader, model, device)
        val_loss.append(avg_loss)

    plot_results(train_loss, val_loss)

def plot_results(train_loss, val_loss):
    plt.plot(train_loss, label='Training Loss')
    plt.plot(val_loss, label='Validation Loss')
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.show() 

def evaluate_model(dataloader, model, device):
    model.eval()
    total_loss = 0
    correct_predictions = 0
    total_samples = 0
    criterion = CrossEntropyLoss()

    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in dataloader:
            image = batch['image'].to(device)
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device)

            outputs = model(input_ids, attention_mask, image)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            correct_predictions += (predicted == labels).sum().item()
            total_samples += labels.size(0)

            predictions.extend(predicted.cpu().numpy())
            true_labels.extend(labels.cpu().numpy())

    accuracy = correct_predictions / total_samples
    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {total_loss / len(dataloader)}, Accuracy: {accuracy}")
    return accuracy, avg_loss

def main():
    train_dataloader, validation_dataset = get_data()
    train_model(train_dataloader, validation_dataset)

if __name__ == '__main__':
    main()