# Pattern Description
This pattern is an example of broadcast followed by a reduction. This type of pattern was recently
used in [OpenFold](https://github.com/aqlaboratory/openfold) model. There are more details
[here](https://github.com/facebookresearch/xformers/pull/160) and [here](https://github.com/pytorch/pytorch/issues/69654). 

If run eagerly, there will be a big
memory allocation just after the broadcast operation. This can make the pattern seem memory
bandwidth bound. However, if the broadcast and reduction operations are fused, we can save the
memory and reduce the memory bandwidth requirements.


TODO - Find representative shapes
