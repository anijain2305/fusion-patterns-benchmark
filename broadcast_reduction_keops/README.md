# Pattern Description
This pattern is an example of broadcast followed by reduction. If run eagerly, there will be a big
memory allocation just after the broadcast operation. This can make the pattern seem memory
bandwidth bound. However, if the broadcast and reduction operations are fused, we can save the
memory and reduce the memory bandwidth requirements.

This pattern is observed in KeOps.

TODO - Add more details about the pattern like which model, representation shapes etc


# How to run?

~~~
python main.py
~~~

This script with benchmark the pattern. Additionally, the forward and backward graphs will also be
saved in forward and backward directory for further debugging.
