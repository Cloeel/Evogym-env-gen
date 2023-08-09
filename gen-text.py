#テキストデータを学習して似たような文章を生成する
#import torchを使って書いてみる

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json

class TextDataset(Dataset):
    def __init__(self, data, word_to_idx):
        self.data = "".join(data)
        self.word_to_idx = word_to_idx

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        text = [self.data[idx]]
        text_indices = [self.word_to_idx[word] for word in text]
        return torch.tensor(text_indices)

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.embedding = nn.Embedding(input_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size)
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

# ハイパーパラメータの設定
input_size = len(dataset.data)  # 入力のボキャブラリーサイズ
hidden_size = 128  # 隠れ層のサイズ
output_size = len(dataset.data)  # 出力のボキャブラリーサイズ
num_epochs = 100
learning_rate = 0.01

# モデルの定義
model = RNN(input_size, hidden_size, output_size)

# 損失関数と最適化手法の定義
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# 学習
for epoch in range(num_epochs):
    for batch_data in dataloader:
        optimizer.zero_grad()
        hidden = model.init_hidden()
        loss = 0
        for word in batch_data[0]:
            output, hidden = model(word, hidden)
            target = word.view(-1)
            loss += criterion(output, target)
        loss.backward()
        optimizer.step()
    print('Epoch: {}, Loss: {:.4f}'.format(epoch+1, loss.item()))
    
# 文字数を100文字とした場合の予測
hidden = model.init_hidden()
input = torch.tensor([[word_to_idx['-']]]) # 最初の文字を'-'とする
predicted = 'H' 
for i in range(1000):
    output, hidden = model(input, hidden)
    _, predicted_word = torch.max(output.data, 1)
    predicted += idx_to_word[predicted_word.item()]
    input = predicted_word
print(predicted)


