import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from PIL import Image
import random

# from moe2 import MoE
from MoE import MixtureOfExperts, DeepMixtureOfExperts

# 可视化结果
import matplotlib.pyplot as plt
import numpy as np

# Set device
print(torch.cuda.is_available())
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Hyperparameters
input_size_expert = 1296
output_size = 10
learning_rate = 0.01
batch_size = 64
num_epochs = 50

#
num_experts = 4
hidden_size_expert1 = 100
hidden_size_expert2 = 20

hidden_size_gating1 = 50
hidden_size_gating2 = 20
k = 8

 


# 自定义平移变换类，用于记录平移参数
class RandomTranslation:
    def __init__(self, max_translate=4):
        self.max_translate = max_translate
        self.last_translate = None

    def __call__(self, img):
        # 生成随机平移参数
        dx = random.randint(-self.max_translate, self.max_translate)
        dy = random.randint(-self.max_translate, self.max_translate)
        self.last_translate = (dx, dy)
        return transforms.functional.affine(img, angle=0, translate=(dx, dy), scale=1, shear=0)

# 随机平移变换
random_translation = RandomTranslation(max_translate=4)
transform = transforms.Compose([
    transforms.Pad(4),
    random_translation,
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # 归一化
])


# 自定义数据集类
class MNISTWithTransformParams(datasets.MNIST):
    def __init__(self, root, train=True, transform=None, download=False):
        super().__init__(root, train=train, transform=transform, download=download)
        self.transform = transform

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # 将灰度图像转换为PIL图像
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        # 获取平移参数
        '''
        self.transform.transforms 是一个包含所有图像变换的列表。
        last_translate 是 RandomTranslation 类的一个属性，用于记录最近一次应用平移变换的参数。这个属性是在 __call__ 方法中设置的。
        '''
        translate_param = self.transform.transforms[1].last_translate

        return img, target, translate_param


# Load MNIST dataset
train_dataset = MNISTWithTransformParams(root='data', 
                               train=True, 
                               transform=transform,
                               download=True)

test_dataset = MNISTWithTransformParams(root='data', 
                              train=False, 
                              transform=transform)

# Data loader
train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)

# 确认数据加载和变换参数记录
for images, labels, params in train_loader:
    print("Images shape:", images.shape)
    print("Labels:", labels)
    print("Translate params:", params)
    break

#models
model = DeepMixtureOfExperts(input_size_expert, hidden_size_expert1, hidden_size_expert2, output_size, num_experts, hidden_size_gating1, hidden_size_gating2, input_size_gating=None ).to(device)

# 尝试加载现有的模型参数
train_new_model = False
try:
    model.load_state_dict(torch.load('model100_20_new.pth'))
    print("模型参数已加载")
except FileNotFoundError:
    print("没有找到现有的模型参数文件，训练新模型")
    train_new_model = True

# 损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate,momentum=0.9)


# 找到所有张量的最大大小
'''
外层循环：遍历 train_loader 和 test_loader，即训练数据和测试数据。
批次循环：对每个批次（batch）中的数据进行解包，获取 params
张量循环：对每个批次中的 params 进行遍历
获取张量大小：获取每个平移参数张量的大小（长度）
最大长度:找到所有平移参数张量中的最大长度。
'''
max_len = max(param.size(0) for batch in [train_loader, test_loader] for _, _, params in batch for param in params)

# 定义一个函数来填充张量
def pad_tensor(tensor, length):
    if tensor.dim() == 0:  # 如果张量没有维度，将其视为标量
        tensor = tensor.unsqueeze(0)
    pad_size = length - tensor.size(0)
    padded_tensor = torch.cat([tensor, torch.zeros(pad_size, dtype=tensor.dtype)])
    return padded_tensor.tolist()

torch.autograd.set_detect_anomaly(True)

if train_new_model:
    # 记录每个 epoch 的 loss 和 accuracy
    loss_list = []
    accuracy_list = []

    # 训练模型并记录门控输出
    gate_outputs1 = []
    gate_outputs2 = []
    final_gate_outputs1 = []
    final_gate_outputs2 = []
    final_labels = []
    translate_params_list = []

    for epoch in range(num_epochs):
        model.train()
        total_main_loss_num = 0
        '''
        train_loader 的每次迭代返回一个包含三个元素的元组 (images, labels, params)。
        enumerate(train_loader)：对 train_loader 进行枚举（enumerate），这将返回一个迭代器，每次迭代会返回一个包含两个元素的元组 (index, data)
        '''
        for i, (images, labels, params) in enumerate(train_loader):
            # print("Params:", params)
            # Reshape images to (batch_size, input_size)
            images = images.reshape(-1, input_size_expert).to(device)
            labels = labels.to(device)
            # print(f'Images shape: {images.shape}, mean: {images.mean().item()}, std: {images.std().item()}')
            # print(f'Labels: {labels}')

            # Forward pass
            outputs, gate_output1, gate_output2 = model(images)
            # print(f'Outputs: {outputs}')
            # print(f'Labels: {labels}')
            main_loss  = criterion(outputs, labels)
            # print(f'Main Loss: {main_loss.item()}')
            
            # Backward and optimize
            '''
            在训练神经网络时，我们需要计算梯度并更新模型的参数。为了避免在每次反向传播时梯度的累积，我们需要在每次参数更新之前清除梯度。
            '''
            optimizer.zero_grad()
            '''
            通过调用 main_loss.backward()，PyTorch 会自动计算损失函数相对于模型参数的梯度，并将结果存储在每个参数的 .grad 属性中。
            '''
            main_loss.backward()

            # 打印每个参数的梯度
            # print(f'Epoch [{epoch+1}/{num_epochs}], Batch [{i+1}/{len(train_loader)}]')
            # for name, param in model.named_parameters():
            #     if param.grad is not None:
            #         print(f'{name} grad: {param.grad.norm().item():.4f}')
            #     else:
            #         print(f'{name} grad: None')

            '''
            optimizer.step() 会遍历优化器中所有的参数，并根据之前计算的梯度和指定的优化算法（如 SGD、Adam 等）来更新每个参数的值。
            '''
            optimizer.step()
                
            total_main_loss_num += main_loss.item()
            # total_main_loss_num += main_loss.item()
            # total_reg_loss_num += reg_loss.item()
            '''
            遍历 params 列表中的每个元素 param，并将每个 param 转换为 NumPy 数组 np.array(param)。
            extend会将生成的新列表中的元素添加到 translate_params_list 的末尾。
            '''
            # print('params[0]', params[0])
            # print('params[1]', params[1])
            dx_padded = pad_tensor(params[0], max_len)
            dy_padded = pad_tensor(params[1], max_len)
            translate_params_list.append([dx_padded, dy_padded])

        # Reset the running total assignment and expert count
        # model.reset_assignments()
            # Test the model
            # In test phase, we don't need to compute gradients (for memory efficiency)

        #模型切换到评估模式并关闭梯度计算。对测试集的数据进行预测
        '''
        在评估模式下，模型中的 dropout 层会关闭，即所有神经元都会参与计算，不会有任何随机丢弃。
        在评估模式下，模型中的 batch normalization 层会使用训练期间计算得到的滑动均值和方差，而不是当前 mini-batch 的均值和方差。
        '''
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():

            for images, labels, params in test_loader:
                '''
                view 是 PyTorch 中的一个方法，用于重新调整张量的形状，但不改变其数据。
                view 方法的参数指定了新的形状。-1 表示这个维度的大小由 PyTorch 自动计算，以确保总元素数量保持不变。
                作用是将 images 张量重新调整为形状 (new_batch_size, input_size_expert)
                '''
                images = images.view(-1, input_size_expert).to(device)
                labels = labels.to(device)
                outputs, gate_output1, gate_output2 = model(images)
                '''
                .data 属性会返回一个包含相同数据的新张量，但不包括梯度信息。
                torch.max 返回指定维度上的最大值以及最大值的索引。dim=1 表示在每一行（即每个样本）上操作
                返回两个张量：第一个张量是每行的最大值（即最大概率），第二个张量是最大值的索引（即预测类别）。
                由于我们只关心预测的类别索引，可以使用 _ 来忽略第一个张量
                '''
                _, predicted = torch.max(outputs.data, 1)
                '''
                更新 total，即标签的总数。
                '''
                total += labels.size(0)
                '''
                计算正确预测的数量，并累加到 correct 中。
                '''
                correct += (predicted == labels).sum().item()
                '''
                numpy只支持在cpu上计算，所以.cpu()
                '''
                final_gate_outputs1.append(gate_output1.cpu().detach().numpy())
                final_gate_outputs2.append(gate_output2.cpu().detach().numpy())
                final_labels.append(labels.cpu().detach().numpy()) 
                for param in params:
                    dx_padded = pad_tensor(params[0], max_len)
                    dy_padded = pad_tensor(params[1], max_len)
                    translate_params_list.append([dx_padded, dy_padded])

        accuracy = 100* correct / total
        avg_main_loss = total_main_loss_num / len(train_loader)

        loss_list.append(avg_main_loss)
        accuracy_list.append(accuracy)
        print(f'Epoch [{epoch+1}/{num_epochs}], Main Loss: {avg_main_loss:.4f},  Accuracy: {accuracy:.2f}%')

        # 在第40个epoch之后禁用 gating function 的 load balance 功能
        if epoch + 1 == 40:
            model.moe1.gating_function.mask_enabled  = False
            model.moe2.gating_function.mask_enabled  = False
            print("Disabled gating function's load balance feature after epoch 45")

    # 保存模型参数
    torch.save(model.state_dict(), 'model100_20_new.pth')
    print("模型参数已保存到 model4.pth 文件")

else:
    # 如果不需要训练，直接加载测试数据
    model.eval()
    final_gate_outputs1 = []
    final_gate_outputs2 = []
    final_labels = []
    translate_params_list = []
    with torch.no_grad():
        for images, labels, params in test_loader:
            images = images.view(-1, input_size_expert).to(device)
            labels = labels.to(device)
            outputs, gate_output1, gate_output2 = model(images)
            final_gate_outputs1.append(gate_output1.cpu().detach().numpy())
            final_gate_outputs2.append(gate_output2.cpu().detach().numpy())


            final_labels.append(labels.cpu().detach().numpy())
            # print(params)
            dx_padded = pad_tensor(params[0], max_len)
            # print('params[0]',params[0])
            # print('dx_padded',dx_padded)
            dy_padded = pad_tensor(params[1], max_len)
            translate_params_list.append([dx_padded, dy_padded])
            # print('translate_params_list:',translate_params_list)

            # print('translate_params_list',translate_params_list)

# print('final_gate_outputs1_shape:', final_gate_outputs1.shape)
final_gate_outputs1 = np.concatenate(final_gate_outputs1, axis=0)

final_gate_outputs2 = np.concatenate(final_gate_outputs2, axis=0)
final_labels = np.concatenate(final_labels, axis=0)
translate_params_list = np.array(translate_params_list)  # 修改这里，转换为 NumPy 数组并转置
print('translate_params_list_shape:', translate_params_list.shape)
# 调整形状，将其从 (157, 2, 64) 转换为 (157*64, 2)
reshaped_params = np.reshape(translate_params_list, (157*64, 2))
# 添加一个新的维度，使其形状为 (64*157, 2, 1)
reshaped_params = np.expand_dims(reshaped_params, axis=2)



# 确保 final_gate_outputs1 和 final_gate_outputs2 的形状正确
print("final_gate_outputs1:", final_gate_outputs1.shape)  # 应该是 (10000, 4, 1)
print("final_gate_outputs2:", final_gate_outputs2.shape)  # 应该是 (10000, 4, 1)
print("translate_params_list:", translate_params_list.shape)  # 应该是 (157, 2, 64)
print("reshaped_params:", reshaped_params.shape) #应该是 (10048, 2, 1)
print("final_labels:", final_labels.shape)#应该是(10000,),一维数组

        # print(f'Accuracy of the model on the 10000 test images: {correct/total} %')
    # eval_dict[epoch] = accuracy
    # eval_dict[i] = evals



import matplotlib.pyplot as plt
import numpy as np

def plot_translation_sensitivity(gate_outputs,translate_params, filename):
    """
    绘制每个专家对图像平移的敏感程度。
    
    参数:
        gate_outputs (numpy array): 形状为 (num_samples, num_experts) 的门控输出。
    """
    num_experts = 4  # 每层4个专家
    num_transforms = 81  # 确定的平移变换数量

    '''
    确保gate_outputs的样本数量是num_transforms的整数倍，以便后续的计算。
    '''
    total_samples = (gate_outputs.shape[0] // num_transforms) * num_transforms
    gate_outputs = gate_outputs[:total_samples]
    print("gate_outputs:", gate_outputs.shape)
    translate_params = translate_params[:total_samples]
    print("translate_params:",translate_params.shape)

    samples_per_transform = total_samples // num_transforms  # 每种变换的样本数量

    fig, axes = plt.subplots(2, 2, figsize=(10, 10))
    for i in range(num_experts):
        #ax 代表当前专家的子图位置。这里通过 i // 2 和 i % 2 来确定子图在 2x2 网格中的位置。
        ax = axes[i // 2, i % 2]
        
        # 计算每个平移变换的平均门控输出
        transform_sensitivity = np.zeros((9, 9))
        #使用 zip 函数将 translate_params 和 gate_outputs[:, i] 进行配对迭代
        for (dx, dy), output in zip(translate_params, gate_outputs[:, i]):
            # print((dx, dy))
            # print('dx[0]',dx[0])
            # print('dx',dx)

            x_idx = int(dx + 4)  # 将dx转换为0到8的索引
            
            y_idx = int(dy + 4)  # 将dy转换为0到8的索引

            #将门控输出 output 累加到对应平移位置的敏感度矩阵中
            transform_sensitivity[y_idx, x_idx] += output

        transform_sensitivity /= samples_per_transform
        
        
        cax = ax.matshow(transform_sensitivity, cmap='jet')
        fig.colorbar(cax, ax=ax)
        ax.set_title(f'expert {i+1}')
        ax.set_xlabel('translation X')
        ax.set_ylabel('translation Y')
        ax.set_xticks(range(9))
        ax.set_yticks(range(9))
        ax.set_xticklabels(range(9))
        ax.set_yticklabels(range(9))
    
    #保存图像
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()


def plot_class_sensitivity(gate_outputs, labels,filename):
    """
    绘制每个专家对MNIST类别的敏感程度。
    
    参数:
        gate_outputs (numpy array): 形状为 (num_samples, num_experts) 的门控输出。
        labels (numpy array): 形状为 (num_samples,) 的真实标签。
    """

    num_experts = 4  # 每层4个专家
    num_classes = 10  # MNIST有10个类别
    
    # 初始化一个数组来存储每个专家和类别的平均门控分配
    class_sensitivity = np.zeros((num_experts, num_classes))
    
    # 计算每个专家和类别的平均门控分配
    for cls in range(num_classes):
        #使用 np.where 函数找到所有标签等于当前类别 cls 的样本索引，返回是一个一维数组
        class_indices = np.where(labels == cls)[0]
        print("gate_outputs[class_indices]_shape:",gate_outputs[class_indices].shape) #应该是(num_class_samples, 4, 1)
        '''
        mean(axis=0)输出在第一个维度（样本维度）上的均值。
        '''
        print('gate_outputs[class_indices].mean(axis=0)_shape:',gate_outputs[class_indices].mean(axis=0).shape) #应该是形状为 (4, 1) 的数组
        class_sensitivity[:, cls] = gate_outputs[class_indices].mean(axis=0).squeeze()
    
    fig, ax = plt.subplots(figsize=(8, 6))
    cax = ax.matshow(class_sensitivity, cmap='jet')
    fig.colorbar(cax)
    ax.set_title('类别敏感性')
    ax.set_xlabel('类别')
    ax.set_ylabel('专家')
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_experts))
    plt.savefig(filename)  # 保存图像
    plt.close()  # 关闭图像





# 可视化第一层门控输出


plot_translation_sensitivity(final_gate_outputs1, reshaped_params, 'layer1_translation.png')
plot_class_sensitivity(final_gate_outputs1, final_labels, 'layer1_class.png')
plot_translation_sensitivity(final_gate_outputs2, reshaped_params, 'layer2_translation.png')
plot_class_sensitivity(final_gate_outputs2, final_labels,'layer2_class.png')

