import torch
import torch.nn as nn
from einops import rearrange 

class GatingFunction(nn.Module):
    def __init__(self, input_size, hidden_size, num_experts, margin_threshold=0.1, enable_mask_before=2):
        super(GatingFunction, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_experts)
        self.margin_threshold = margin_threshold
        self.enable_mask_before = enable_mask_before
        self.mask_enabled = True  # 添加这个控制变量

        # 初始化 running total assignment 和 expert count
        '''
        添加缓冲区，
        running_total_assignment：每个专家的累计分配总和，
        expert_count：记录处理过的总样本数。
        作用：注册一个不可训练的张量（buffer）。这些 buffer 不会在反向传播时计算梯度，也不会通过优化器更新。它们通常用于保存模型的状态信息
        '''
        self.register_buffer('running_total_assignment', torch.zeros(num_experts))
        self.register_buffer('sample_count', torch.zeros(1))

    def forward(self, x, training = True):
        '''
        它接收输入并通过两个全连接层（带激活函数ReLU）来计算每个专家的选择概率。
        softmax函数确保输出是概率分布。
        '''
        x = self.fc1(x)
        x = torch.relu(x)
        # print(f'GatingFunction fc1 output: {x}')
        x = self.fc2(x)
        x = torch.softmax(x, dim=1)
        # print(f'GatingFunction fc2 output (after softmax): {x}')

        if training and self.mask_enabled:
            
            margin = self.margin_threshold

            # 初始化一个与x形状相同的张量来存储修改后的结果
            modified_x = x.clone()  # 深度拷贝x，以免修改原x

            
            # 对每个专家进行约束，mask-->bool
            for i in range(x.size(0)):
                sample = x[i]

                # Step 1: Add the sample to running_total_assignment
                #detach()分理处张量，防止反向传播通过
                self.running_total_assignment += sample.detach()
                
                # Step 2: Calculate mean_assignment and mask
                mean_assignment = self.running_total_assignment.mean()
                mask = self.running_total_assignment > (mean_assignment + margin)
                # print("mean_assignment:",mean_assignment)
                # print("mask:", mask)
                

                # Step 3: If mask has True, subtract the sample added earlier
                if mask.any():
                    self.running_total_assignment -= sample.detach()
                    
                # #Debugging: 打印每个样本的mask
                # print(f"Sample {i}: {sample}")
                # print(f"Mask for sample {i}: {mask}")
                # Step 4: Apply mask to the sample and normalize
                modified_sample  = sample.clone().detach()
                modified_sample[mask] = 0

                normalization_factor=modified_sample.sum()
                if normalization_factor == 0:
                    normalization_factor = 1
                modified_sample = modified_sample / normalization_factor
                # print("modified_sample:",modified_sample)

                # Step 5: Update modified_x and running_total_assignment
                modified_x[i] = modified_sample
                self.running_total_assignment += modified_sample.detach()
                #Debugging: Print the normalized output
                # print(f'Normalized output: {modified_x}')
            
            
            x = modified_x
            # print("x:",x)

        return x

class Expert(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Expert, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        
        # nn.init.kaiming_normal_(self.fc1.weight)
        # nn.init.zeros_(self.fc1.bias)
        # nn.init.kaiming_normal_(self.fc2.weight)
        # nn.init.zeros_(self.fc2.bias)
    def forward(self, x):
        '''
        每个专家是一个简单的两层全连接神经网络。
        '''
        x = self.fc1(x)
        x = torch.relu(x)
        # print(f'Expert fc1 output: {x}')
        x = self.fc2(x)
        # print(f'Expert fc2 output: {x}')
        return x

class MixtureOfExperts(nn.Module):
    '''
    input_size_expert：每个专家的输入大小。
    hidden_size_expert：每个专家的隐藏层大小。
    output_size：每个专家的输出大小。
    num_experts：专家的数量。
    input_size_gating 和 hidden_size_gating：门控函数的输入和隐藏层大小，如果未提供，则默认与专家的输入和隐藏层大小相同。
    '''
    def __init__(self, input_size_expert, hidden_size_expert, output_size, num_experts, input_size_gating=None, hidden_size_gating=None):
        super(MixtureOfExperts, self).__init__()
        if input_size_gating is None:
            input_size_gating = input_size_expert
        if hidden_size_gating is None:
            hidden_size_gating = hidden_size_expert
        self.gating_function = GatingFunction(input_size_gating, hidden_size_gating, num_experts)
        self.experts = nn.ModuleList([Expert(input_size_expert, hidden_size_expert, output_size) for _ in range(num_experts)])
        
    def forward(self, x_expert, x_gating=None):
        '''
        门控函数输入：如果未提供 x_gating，则使用 x_expert 作为输入。
        计算门控输出：使用门控函数计算每个专家的权重。
        '''
        if x_gating is None:
            x_gating = x_expert
        gate_output = self.gating_function(x_gating)


        # print(f'Gate outputs before checking: {gate_output}')
    
        '''
        expert(x_expert) 计算出每个专家的输出
        unsqueeze(1) 将输出在第1维增加一个维度
        形状变为 (batch_size, 1, output_size
        '''
        expert_outputs = [expert(x_expert).unsqueeze(1) for expert in self.experts]
        '''
        连接后的 expert_outputs 形状为 (batch_size, num_experts, output_size)。
        '''
        expert_outputs = torch.cat(expert_outputs, dim=1)
        '''
        最后一维增加一个维度，使得 gate_output 的形状变为 (batch_size, num_experts, 1)，以便在与 expert_outputs 进行逐元素相乘时进行广播。
        '''
        gate_output = gate_output.unsqueeze(-1) 
        '''
        对门控输出和专家输出进行逐元素相乘，形状为 (batch_size, num_experts, output_size)
        在第1维上求和，将所有专家的加权输出求和，得到最终输出 output。此时，output 的形状为 (batch_size, output_size)
        '''
        output = torch.sum(gate_output * expert_outputs, dim=1)

        if torch.all(gate_output == 0):
            print("Warning: All gate outputs are zero. Adjusting gate outputs to prevent zero output.")
            gate_output = torch.ones_like(gate_output) / gate_output.size(1)
        
        # print(f'Expert outputs before gating: {expert_outputs}')
        # print(f'Gate outputs: {gate_output}')
        output = torch.sum(gate_output * expert_outputs, dim=1)
        # print(f'Final output after gating: {output}')
        return output, gate_output


# 定义两层MixtureOfExperts模型
class DeepMixtureOfExperts(nn.Module):
    def __init__(self, input_size_expert, hidden_size_expert1, hidden_size_expert2, output_size, num_experts, hidden_size_gating1=None, hidden_size_gating2 = None, input_size_gating=None):
        super(DeepMixtureOfExperts, self).__init__()
        self.moe1 = MixtureOfExperts(input_size_expert, hidden_size_expert1, output_size, num_experts,input_size_gating, hidden_size_gating1)
        self.moe2 = MixtureOfExperts(output_size, hidden_size_expert2, output_size, num_experts,input_size_gating, hidden_size_gating2)

    def forward(self, x):
        x, gate_output1 = self.moe1(x)
        # print(f'MOE1 output: {x}')
        x, gate_output2 = self.moe2(x)
        # print(f'MOE2 output: {x}')
        return x, gate_output1, gate_output2


