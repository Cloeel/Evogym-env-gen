import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

class TextDataset(Dataset):
    def __init__(self, data, word_to_idx):
        self.data = "".join(data)
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = self.data[idx]
        text_indices = [self.word_to_idx[word] for word in text]
        return torch.tensor(text_indices)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(output_size, hidden_size)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded
        output, hidden = self.gru(output, hidden)
        output = self.fc(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

# JSONファイルの読み込みとデータの前処理
with open('./json-list-data/CaveCrawler-v0.json', 'r') as f:
    data = json.load(f)

# テキストデータの前処理とボキャブラリー作成
words = [word for text in data for word in text]
vocab = list(set(words))
word_to_idx = {word: i for i, word in enumerate(vocab)}
idx_to_word = {i: word for i, word in enumerate(vocab)}

# データセットの作成
dataset = TextDataset(data, word_to_idx)
dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
print(dataloader)

# ハイパーパラメータの設定
input_size = len(dataset.data)  # 入力のボキャブラリーサイズ
hidden_size = 128  # 隠れ層のサイズ
output_size = len(dataset.data)  # 出力のボキャブラリーサイズ
num_epochs = 100
learning_rate = 0.001
print(input_size)

# モデルの初期化
model = RNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# トレーニングループ
for epoch in range(num_epochs):
    loss = 0
    hidden = model.init_hidden()
    for batch in dataloader:
        text = batch.squeeze()
        print(batch)
        target = text[1:]
        text_input = text[:-1]

        # モデルの出力と損失の計算
        output, hidden = model(text_input, hidden)
        loss += criterion(output.squeeze(), target)

        # バックプロパゲーションとパラメータの更新
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    # エポックごとの損失の表示
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item()}")

# 生成のテスト
start_letter = 'T'
with torch.no_grad():
    input = torch.tensor([[word_to_idx[start_letter]]])
    hidden = model.init_hidden()
    output_text = start_letter

    # 生成されたテキストの最大長を指定
    max_length = 100

    # モデルを使ってテキストを生成
    for i in range(max_length):
        output, hidden = model(input, hidden)
        _, predicted = torch.max(output, 2)
        output_text += idx_to_word[predicted.item()]
        input = predicted.view(1, 1)

    print(output_text)