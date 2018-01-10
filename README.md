# HPGMG-FV CUDA Async

[HPGMG](https://hpgmg.org) is an HPC benchmarking effort and supercomputing ranking metric based on geometric multigrid methods. NVIDIA reworked the original implementation moving from a CPU implementation to an [hybrid CPU-GPU solution](https://bitbucket.org/nsakharnykh/hpgmg-cuda).
In this repository, starting from the NVIDIA solution, we leverage the communications with [GPUDirect Async](https://github.com/gpudirect/libgdsync), recently released by NVIDIA.

For further information about Async and benchmarks, please refer to:

 - ["GPUDirect Async: exploring GPU synchronous communication techniques for InfiniBand clusters"](https://www.sciencedirect.com/science/article/pii/S0743731517303386), E. Agostini, D. Rossetti, S. Potluri. Journal of Parallel and Distributed Computing, Vol. 114, Pages 28-45, April 2018
 - ["Offloading communication control logic in GPU accelerated applications"](http://ieeexplore.ieee.org/document/7973709), E. Agostini, D. Rossetti, S. Potluri. Proceedings of the 17th IEEE/ACM International Symposium on Cluster, Cloud and Grid Computing (CCGrid’ 17), IEEE Conference Publications, Pages 248-257, Nov 2016
 