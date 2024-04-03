import torch
import torch.nn as nn
import torch.nn.functional as F
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
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)  # 这里的batch_size根据实际情况调整
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
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
            #print(outputs.shape, structures.shape)
            #print(outputs, structures)
            loss = criterion(outputs, structures)
            loss.backward()
            optimizer.step()
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item()}')

# 模型参数
input_size = 20  # 假设有20种不同的氨基酸
hidden_size = 64  # 隐藏层大小
output_size = 3  # 有3种二级结构类别

model = SimpleNN(input_size, hidden_size, output_size)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.01, weight_decay=0.01)

# 训练和评估模型

#device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
train_model(model, train_loader, criterion, optimizer, num_epochs=20)


model.eval()  # 设置模型为评估模式
correct_predictions = 0
total_samples = 0

with torch.no_grad():  # 关闭梯度计算
    for sequences, structures in test_loader:
        outputs = model(sequences)
        _, predicted = torch.max(outputs, 1)  # 获取预测的类别
        correct_predictions += (predicted == structures).sum().item()
        total_samples += structures.size(0)
        #print(f'Predicted: {predicted}, Structures: {structures}')
        print(f'Correct predictions: {correct_predictions}, Total samples: {total_samples}')
accuracy = correct_predictions / total_samples * 100
print(f'Accuracy: {accuracy}%')
# 这里仅展示模型构建、训练和评估的代码









