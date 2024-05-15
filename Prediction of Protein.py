import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence

# 定义氨基酸和二级结构的编码
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
secondary_structures = '_eh'

amino_to_idx = {amino: idx for idx, amino in enumerate(amino_acids)}
struct_to_idx = {struct: idx for idx, struct in enumerate(secondary_structures)}


def encode_sequence(sequence):
    return [amino_to_idx[amino] for amino in sequence]


def encode_structure(structure):
    return [struct_to_idx[struct] for struct in structure]


# 自定义数据集类
class ProteinDataset(Dataset):
    def __init__(self, filepath):
        self.data = []
        self.labels = []
        with open(filepath, 'r') as f:
            sequence = []
            label = []
            for line in f:
                if line.strip() == "<end>":
                    if sequence:
                        self.data.append(torch.tensor(encode_sequence(sequence), dtype=torch.long))
                        self.labels.append(torch.tensor(encode_structure(label), dtype=torch.long))
                    sequence = []
                    label = []
                else:
                    parts = line.split()
                    if len(parts) == 2:
                        amino, sec_struct = parts
                        sequence.append(amino)
                        label.append(sec_struct)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


# 自定义 collate_fn 函数
def collate_fn(batch):
    sequences, structures = zip(*batch)
    padded_sequences = pad_sequence(sequences, batch_first=True, padding_value=0)
    padded_structures = pad_sequence(structures, batch_first=True, padding_value=-1)
    return padded_sequences, padded_structures


# 加载数据集
train_dataset = ProteinDataset(
    r'C:\Users\11932\Desktop\Github\Protein-Prediction\protein-secondary-structure.train - 副本.txt')
test_dataset = ProteinDataset(
    r'C:\Users\11932\Desktop\Github\Protein-Prediction\protein-secondary-structure.test - 副本.txt')

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)


# 定义神经网络模型
class ProteinNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, embedding_dim):
        super(ProteinNN, self).__init__()
        self.embedding = nn.Embedding(input_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=2)  # 输出的 softmax 应该在最后一个维度上进行

    def forward(self, x):
        x = self.embedding(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.softmax(x)
        return x

# 训练模型的函数
def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        model.train()
        for sequences, structures in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            outputs = outputs.view(-1, output_size)
            structures = structures.view(-1)
            loss = criterion(outputs, structures)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


# 评估模型的函数
def evaluate_model(model, data_loader, criterion=None):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        loss_sum = 0
        for sequences, structures in data_loader:
            outputs = model(sequences)
            outputs = outputs.view(-1, output_size)
            structures = structures.view(-1)
            _, predicted = torch.max(outputs.data, 1)
            mask = (structures != -1)
            total += mask.sum().item()
            correct += (predicted == structures).masked_select(mask).sum().item()
            if criterion:
                loss_sum += criterion(outputs, structures).item()
    accuracy = 100 * correct / total
    if criterion:
        avg_loss = loss_sum / len(data_loader)
        return avg_loss
    return accuracy


# 模型参数
input_size = 20  # 氨基酸种类数
hidden_size = 50  # 隐藏层大小
output_size = 3  # 二级结构种类数
embedding_dim = 10  # 嵌入维度

model = ProteinNN(input_size, hidden_size, output_size, embedding_dim)
criterion = nn.NLLLoss(ignore_index=-1)  # 忽略填充值
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 训练和评估模型
num_epochs = 30  # 训练的轮数
train_model(model, train_loader, criterion, optimizer, num_epochs)

# 评估训练集和测试集上的准确率
train_accuracy = evaluate_model(model, train_loader)
test_accuracy = evaluate_model(model, test_loader)

print(f'Accuracy on training set: {train_accuracy:.2f}%')
print(f'Accuracy on test set: {test_accuracy:.2f}%')
