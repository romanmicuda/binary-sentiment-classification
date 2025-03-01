import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from collections import Counter
import numpy as np
import matplotlib.pyplot as plt
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# [SentimentDataset and SentimentClassifier classes remain unchanged]
class SentimentDataset(Dataset):
    def __init__(self, comments, labels, word2idx, max_len=50):
        self.comments = comments
        self.labels = labels
        self.word2idx = word2idx
        self.max_len = max_len
        
    def __len__(self):
        return len(self.comments)
    
    def __getitem__(self, idx):
        comment = str(self.comments[idx]).lower().split()
        sequence = [self.word2idx.get(word, 0) for word in comment[:self.max_len]]
        if len(sequence) < self.max_len:
            sequence = sequence + [0] * (self.max_len - len(sequence))
        
        return {
            'text': torch.tensor(sequence, dtype=torch.long),
            'label': torch.tensor(self.labels[idx], dtype=torch.float)
        }

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim=100, hidden_dim=256, output_dim=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, text):
        embedded = self.embedding(text)
        output, (hidden, cell) = self.lstm(embedded)
        return self.sigmoid(self.fc(hidden[-1]))

def prepare_data(df):
    df = df[df['Sentiment'] != 'neutral'].dropna()
    le = LabelEncoder()
    labels = le.fit_transform(df['Sentiment'])
    all_words = ' '.join(df['Comment']).lower().split()
    word_counts = Counter(all_words)
    vocab = ['<PAD>'] + [word for word, count in word_counts.items() if count > 1]
    word2idx = {word: idx for idx, word in enumerate(vocab)}
    return df['Comment'].values, labels, word2idx, len(vocab)

def train_model(model, train_loader, val_loader, epochs=10, learning_rate=0.00003, patience=10):
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    train_losses = []
    val_losses = []
    val_accuracies = []
    
    model = model.to(device)
    criterion = criterion.to(device)
    
    best_val_loss = float('inf')
    epochs_no_improve = 0
    
    for epoch in range(epochs):
        model.train()
        total_train_loss = 0
        for batch in train_loader:
            text = batch['text'].to(device)
            labels = batch['label'].to(device)
            
            optimizer.zero_grad()
            predictions = model(text).squeeze(1)
            loss = criterion(predictions, labels)
            loss.backward()
            optimizer.step()
            total_train_loss += loss.item()
        
        model.eval()
        val_loss = 0
        correct = 0
        total = 0
        with torch.no_grad():
            for batch in val_loader:
                text = batch['text'].to(device)
                labels = batch['label'].to(device)
                
                predictions = model(text).squeeze(1)
                val_loss += criterion(predictions, labels).item()
                predicted = (predictions >= 0.5).float()
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = correct / total
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        val_accuracies.append(val_accuracy)
        
        print(f'Epoch {epoch+1}:')
        print(f'Train Loss: {avg_train_loss:.4f}')
        print(f'Val Loss: {avg_val_loss:.4f}')
        print(f'Val Accuracy: {val_accuracy:.4f}')
        
        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        
        if epochs_no_improve == patience:
            print(f'Early stopping triggered after {epoch+1} epochs.')
            break
    
    history_df = pd.DataFrame({
        'Epoch': range(1, len(train_losses) + 1),
        'Train_Loss': train_losses,
        'Val_Loss': val_losses,
        'Val_Accuracy': val_accuracies
    })
    
    return history_df

def plot_training_history(history_df):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    ax1.plot(history_df['Epoch'], history_df['Train_Loss'], label='Training Loss')
    ax1.plot(history_df['Epoch'], history_df['Val_Loss'], label='Validation Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history_df['Epoch'], history_df['Val_Accuracy'], label='Validation Accuracy', color='green')
    ax2.set_title('Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig('training_history.png')


def main(df):
    
    comments, labels, word2idx, vocab_size = prepare_data(df)
    
    X_train, X_test, y_train, y_test = train_test_split(
        comments, labels, test_size=0.2, random_state=42
    )
    
    train_dataset = SentimentDataset(X_train, y_train, word2idx)
    test_dataset = SentimentDataset(X_test, y_test, word2idx)
    
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    model = SentimentClassifier(vocab_size=vocab_size).to(device)
    
    history_df = train_model(model, train_loader, test_loader, epochs=100)
    
    print("\nTraining History DataFrame:")
    print(history_df)
    
    plot_training_history(history_df)
    
    # Save the model and preprocessing data
    torch.save(model.state_dict(), 'sentiment_model.pth')
    with open('word2idx.pkl', 'wb') as f:
        pickle.dump(word2idx, f)
    print("\nModel and word2idx saved successfully!")

if __name__ == "__main__":
    df = pd.read_csv('data/YoutubeCommentsDataSet.csv')
    main(df)