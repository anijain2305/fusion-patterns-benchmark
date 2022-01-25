# Pattern Description
This pattern is an example of upsampling, non linearity and down sample operation. This type of
patterns are common in StyleGans. PyTorch eager allocates a large memory chunk in between the ops.
Operator fusion can lead to big savings here.

# How to run?

~~~
python main.py
~~~

This script with benchmark the pattern. Additionally, the forward and backward graphs will also be
saved in generated forward and backward python files for further debugging.
