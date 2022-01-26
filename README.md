# Fusion Pattern Benchmark
Here, we collect commonly observed operators or sequence of operators in research and production.
Our goal is to benchmark these patterns for PyTorch eager and different compilers, like Torchscript
and TVM.

Each pattern is present in a separate directory.


# How to run?
~~~

python main.py # Runs all the benchmarks
python main.py --p=outer_product_mean # Runs a particular pattern
~~~

The script will also save the forward and backward graphs in the relevant pattern directory. These
can be further used for debugging/visualization.
