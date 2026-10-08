# 笔试专项：Attention 维度、Mask 与 KV Cache

## Self-Attention

输入 `X` 的形状为 `[B, L, d_model]`。线性投影 `Q=XW_Q`、`K=XW_K`、`V=XW_V`；若 `h` 个头、每头 `d_k=d_model/h`，分头后常见形状为 `[B, h, L, d_k]`。

```text
scores = Q K^T / sqrt(d_k)             # [B, h, L, L]
weights = softmax(scores + mask)        # [B, h, L, L]
context = weights V                     # [B, h, L, d_k]
```

这里转置的是最后两个维度。除以 `sqrt(d_k)` 能减小点积尺度，避免 softmax 过度饱和。输出各头拼接后再经投影回 `[B,L,d_model]`。

- **Padding mask**：屏蔽补齐位置，防止真实 token 关注 padding。
- **Causal mask**：屏蔽当前位置之后的 token，用于自回归生成。
- BERT 常用双向编码和掩码语言建模；GPT 常用因果解码和下一个 token 预测。具体模型变体仍要看结构与训练目标。

**维度题**：`B=2、L=8、d_model=64、h=4` 时，每头维度 `d_k=16`，Q/K/V 为 `[2,4,8,16]`，分数矩阵为 `[2,4,8,8]`。

## KV Cache

自回归生成分为 prefill（处理输入上下文）和 decode（逐 token 生成）。decode 时缓存各层历史 token 的 K/V，新 token 仍需计算自己的 Q/K/V，并让新 Q 与历史及当前 K 做注意力，随后把新 K/V 追加到缓存。这样避免反复计算历史 token 的 K/V，但注意力仍要读取历史 K/V，显存随上下文长度、层数及并发增长。

近似缓存元素数：`2 × 层数 × B × L × KV头数 × 每头维度`；字节数再乘 dtype 字节数。MHA 的 KV 头数通常等于查询头数；MQA/GQA 减少 KV 头数，可降低缓存占用和读取量。不要说 KV Cache 让每个新 token 的注意力计算完全变成常数。

## 检查题

1. 为什么缩放因子是 `sqrt(d_k)`？
2. Q/K/V 分头后各是什么形状？`QK^T` 呢？
3. Padding mask 和 causal mask 分别阻止什么？
4. KV Cache 节省了什么计算，又增加了什么资源消耗？
