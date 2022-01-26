# Pattern Description
This pattern is an example of upsampling, non linearity and down sample operation. Specifically, the
pattern is taken from the StyleGan3 research. Authors mentions in the
[paper](https://nvlabs-fi-cdn.nvidia.com/stylegan3/stylegan3-paper.pdf) that their hand-written CUDA
implementation for this pattern was 20-40x faster, leading to 10x improvement in training time.
Authors have open sourced their implementation
[here](https://github.com/NVlabs/stylegan3/tree/a5a69f58294509598714d1e88c9646c3d7c6ec94).
Specifically, the kernel of interest is
[this](https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/torch_utils/ops/filtered_lrelu.py#L1).
We look at their [reference
implementation](https://github.com/NVlabs/stylegan3/blob/a5a69f58294509598714d1e88c9646c3d7c6ec94/torch_utils/ops/filtered_lrelu.py#L121)
to generate this pattern.

We find the representative shapes from their paper
Input size = 512 x 512 x 32
Filter size n = 6
