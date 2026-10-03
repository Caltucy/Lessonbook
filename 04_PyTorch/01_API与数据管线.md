# PyTorch API 与数据管线

核心组件：

- `torch.Tensor`：保存数据和梯度计算信息。
- `Dataset`：定义如何按索引读取一个样本。
- `DataLoader`：批量加载、打乱和并行读取数据。
- `nn.Module`：定义模型结构和可学习参数。
- `loss_fn`：把预测和标签变成标量损失。
- `optimizer`：根据梯度更新参数。

常见排查顺序：打印输入和标签的 shape，检查 dtype、类别编号、device，以及模型输出是否与损失函数要求一致。
