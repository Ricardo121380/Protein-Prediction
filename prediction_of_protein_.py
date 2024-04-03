import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd

# 假设氨基酸序列使用字母A-Z表示（实际上只有20种氨基酸），二级结构用'e', 'h', '_'表示
amino_acids = 'ACDEFGHIKLMNPQRSTVWY'
structures = 'eh_'

# 创建one-hot编码字典
aa_to_onehot = {aa: np.eye(len(amino_acids), dtype=np.float32)[i] for i, aa in enumerate(amino_acids)}
structure_to_index = {s: i for i, s in enumerate(structures)}


class ProteinDataset(Dataset):
    def __init__(self, filepath):
        self.sequences, self.structures = self.load_data(filepath)

    def load_data(self, filepath):
        sequences = []
        structures = []
        with open(filepath, 'r') as file:
            for line in file:
                parts = line.strip().split()
                if len(parts) == 2:
                    seq, struct = parts
                    sequences.append([aa_to_onehot[aa] for aa in seq])
                    structures.append([structure_to_index[s] for s in struct])
        return np.array(sequences), np.array(structures)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        # 将数据转换为Tensor
        sequence_tensor = torch.tensor(self.sequences[idx], dtype=torch.float32)
        structure_tensor = torch.tensor(self.structures[idx], dtype=torch.long)  # 对于CrossEntropyLoss，target需要是long类型
        return sequence_tensor, structure_tensor


# 示例：加载数据集
train_dataset = ProteinDataset(r'C:\Users\11932\Desktop\Protein Prediction\protein-secondary-structure.train - 副本.txt')
test_dataset = ProteinDataset(r'C:\Users\11932\Desktop\Protein Prediction\protein-secondary-structure.test - 副本.txt')

# 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)  # 这里的batch_size根据实际情况调整
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)


# 这里仅展示模型构建、训练和评估的代码

class ProteinDataset(Dataset):
    def __init__(self, sequences, structures):
        self.sequences = sequences  # 已经转换为one-hot编码
        self.structures = structures  # 已经转换为整数标签

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.structures[idx]


class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        #x = torch.squeeze(x, 1)  # 移除第二个维度（索引为1的维度），如果它的大小是1
        return x


def train_model(model, train_loader, criterion, optimizer, num_epochs):
    for epoch in range(num_epochs):
        for sequences, structures in train_loader:
            optimizer.zero_grad()
            outputs = model(sequences)
            # Reshape the structures tensor if necessary
            if len(structures.shape) > 1:
                structures = structures.squeeze()
            outputs = outputs.squeeze()
            #print(outputs.shape)
            #print(structures.shape)
            loss = criterion(outputs, structures)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')


def evaluate_model(model, test_loader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for sequences, structures in test_loader:
            outputs = model(sequences)
            _, predicted = torch.max(outputs.data, 1)
            total += structures.size(0)
            correct += (predicted == structures).sum().item()
    print(f'Accuracy of the model on the test set: {100 * correct / total}%')


# 模型参数
input_size = 20  # 假设有20种不同的氨基酸
hidden_size = 16  # 隐藏层大小
output_size = 3  # 有3种二级结构类别

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

# 训练和评估模型
# 假设train_dataset和test_dataset已经准备好
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

train_model(model, train_loader, criterion, optimizer, num_epochs=20)
evaluate_model(model, test_loader)



