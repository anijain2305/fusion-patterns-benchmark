# Pattern Description
This pattern is an example of broadcast followed by a reduction. This type of pattern was recently
used in OpenFold model. There are more details here and here. 

If run eagerly, there will be a big
memory allocation just after the broadcast operation. This can make the pattern seem memory
bandwidth bound. However, if the broadcast and reduction operations are fused, we can save the
memory and reduce the memory bandwidth requirements.


TODO - Find representative shapes

# How to run?

~~~
python pattern.py
~~~

This script with benchmark the pattern. Additionally, the forward and backward graphs will also be
saved in forward and backward directory for further debugging.
