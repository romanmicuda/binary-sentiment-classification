import torch
import torch.nn as nn
import pickle

# Device configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model class (must match the training script)
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

# Function to preprocess new comments
def preprocess_comment(comment, word2idx, max_len=50):
    comment = str(comment).lower().split()
    sequence = [word2idx.get(word, 0) for word in comment[:max_len]]
    if len(sequence) < max_len:
        sequence = sequence + [0] * (max_len - len(sequence))
    return torch.tensor(sequence, dtype=torch.long).unsqueeze(0)  # Add batch dimension

# Inference function
def predict_sentiment(model, comment, word2idx, device):
    model.eval()
    with torch.no_grad():
        input_tensor = preprocess_comment(comment, word2idx).to(device)
        prediction = model(input_tensor).squeeze().item()
        sentiment = "positive" if prediction >= 0.5 else "negative"
        confidence = prediction if prediction >= 0.5 else 1 - prediction
        return sentiment, confidence

def main():
    # Load the saved model and word2idx
    with open('word2idx.pkl', 'rb') as f:
        word2idx = pickle.load(f)
    
    vocab_size = len(word2idx)
    model = SentimentClassifier(vocab_size=vocab_size)
    model.load_state_dict(torch.load('sentiment_model.pth'))
    model.to(device)
    print("Model loaded successfully!")

    # Example new comments for inference
    new_comments = [
        "I love how easy this app makes everything!",
        "The user interface is confusing and frustrating",
        "Pretty good experience overall"
    ]

    # Make predictions
    print("\nPredictions:")
    for comment in new_comments:
        sentiment, confidence = predict_sentiment(model, comment, word2idx, device)
        print(f"Comment: {comment}")
        print(f"Sentiment: {sentiment} (Confidence: {confidence:.4f})\n")

if __name__ == "__main__":
    main()