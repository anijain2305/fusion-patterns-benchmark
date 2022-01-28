# Pattern Description
Masked softmax is common in Attention layer. PyTorch core has added a manually fused operator based
on evidences that fused operation is around 5% faster. One thing to note here is that the mask here
is not completely random in real workloads, so there might be a possibility to write/generate faster
code. However, as of now, this seems to be discouraged given the simplicity of current situation.
