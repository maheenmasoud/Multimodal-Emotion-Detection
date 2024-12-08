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

class MultimodalEmotionModel(torch.nn.Module):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # Image branch
        self.image_branch = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(32 * 32 * 32, 256),
            nn.ReLU()
        )

        # Text branch
        self.text_branch = DistilBertModel.from_pretrained('distilbert-base-uncased')
        self.text_fc = nn.Linear(768, 256)

        # Combined branch
        self.fc = nn.Sequential(
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 7)  # Assuming 7 emotion classes
        )

    def forward(self, image, input_ids, attention_mask):
        # Image path
        image_features = self.image_branch(image)

        # Text path
        text_features = self.text_branch(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0, :]
        text_features = self.text_fc(text_features)

        # Combine
        combined = torch.cat((image_features, text_features), dim=1)
        output = self.fc(combined)

        return output

class MultimodalDataset(Dataset):
    def __init__(self, data, transform, tokenizer):
        self.data = data
        self.tokenizer = tokenizer
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        text = row['Utterance']
        label = row['Emotion']
        image = Image.open(row['image_path']).convert('RGB')

        # Process image
        image = self.transform(image)

        # Process text
        input_ids, attention_mask = preprocess_text(text)

        # Convert label to tensor
        label = torch.tensor(label, dtype=torch.long)

        return {'image': image, 'input_ids': input_ids, 'attention_mask': attention_mask, 'label': label}
    
# Image preprocessing
transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

# Text preprocessing
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')

def get_data():
    data = pd.read_csv('sampleData.csv')
    dataset = MultimodalDataset(data, transform, tokenizer)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    return dataloader

def preprocess_text(text):
    encoded = tokenizer(text, padding='max_length', max_length=50, truncation=True, return_tensors="pt")
    return encoded['input_ids'].squeeze(0), encoded['attention_mask'].squeeze(0)

def train_model(dataloader, model, optimizer, criterion, device):
    model.train()
    for epoch in range(5):
        total_loss = 0
        for batch in dataloader:
            images, input_ids, attention_mask, labels = [x.to(device) for x in batch]
            
            optimizer.zero_grad()
            outputs = model(images, input_ids, attention_mask)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        print(f"Epoch {epoch+1}, Loss: {total_loss / len(dataloader)}")

def evaluate_model(dataloader, model, device):
    model.eval()
    all_preds, all_labels = [], []
    
    with torch.no_grad():
        for batch in dataloader:
            images, input_ids, attention_mask, labels = [x.to(device) for x in batch]
            outputs = model(images, input_ids, attention_mask)
            preds = torch.argmax(outputs, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    acc = accuracy_score(all_labels, all_preds)
    cm = confusion_matrix(all_labels, all_preds)
    print(f"Accuracy: {acc}")
    print(f"Confusion Matrix:\n{cm}")


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = MultimodalEmotionModel().to(device)
    optimizer = Adam(model.parameters(), lr=1e-3)
    criterion = CrossEntropyLoss()
    dataloader = get_data()
    train_model(dataloader, model, optimizer, criterion, device)
    evaluate_model(dataloader, model, device)


if __name__ == '__main__':
    main()