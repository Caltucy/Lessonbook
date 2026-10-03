# Transformer 与 Embedding

Transformer 通过 Self-Attention 建立 token 之间的关系，再经过前馈网络、残差连接和归一化进行表示变换。Embedding 是把离散 token 映射为连续向量，向量空间中的距离可以用于相似度检索。

需要能区分：

- 预训练、推理和微调
- Prompt、RAG 和参数微调
- Token 数量、上下文窗口和推理成本
