# Pattern Description
This pattern is an example of upsampling, non linearity and down sample operation. This type of
patterns are common in StyleGans. PyTorch eager allocates a large memory chunk in between the ops.
Operator fusion can lead to big savings here.
