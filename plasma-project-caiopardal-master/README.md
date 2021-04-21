MO644/MC970 - Final Project
================================================================================

This is the base repository for PLASMA to parallelize using the
[OmpCluster][ompcluster] programming model. In order to
keep all the projects standardized, you will be required to follow some
conventions. Please read carefully the instructions below.

Structure
--------------------------------------------------------------------------------

The PLASMA project is organized using the following directory structure, do not
change it:

- **Source Code**: The code is divided into three different folders:
  + ***test***: This directory contains the code of the different applications
  implemented in PLASMA. Look in this folder to find the assigned application.
  For example, there are applications resolving linear systems of equations,
  least squares problems, eigenvalue problems, and singular value problems.
  + ***compute***: This directory contains the code that launches each
  application, manages the parameter and verifies the validity of the output
  data provided to the application and calls the kernels. It is also where the
  OpenMP parallel region is initialized. All the applications assigned for the
  final project are using more than one kernel.
  + ***core_blas***: This directory contains the different kernels that are used
  by the applications implemented in the test folder. The initial
  parallelization with `#pragma omp task` and the calls to the BLAS
  functions in charge of performing the calculation are done in this folder.

- **Build**: Use the following commands to build PLASMA:
```
export CC=clang
cmake ..
make -j4
```
You should use CMake 3.13 to generate the build system and the OmpCluster fork
of Clang to compile the code. They are both available in the docker image
provided for the project `ompcluster/plasma-dev:latest`. See below for more
details on how to use it.

- **Execution**: The following command line execute the program:
```
./plasmatest $application_name
```
For example, the *dgeadd* application can be executed using:
```
./plasmatest dgeadd
```
The result will be:
```
Status      Error       Time       Gflop/s      transA  transB  m       n       k     nb    alpha           beta            padA  padB  padC

    pass   0.00e+00     0.0008     0.0002       n       n       1000    1000    1000  256   1.23 +  2.35i   6.79 +  7.89i     0     0     0
```
Where:

- **Status** determines if the parallelization code has similar results with
serial code.
- **Erro** is the difference between the result obtained from the parallelized
code and the serial code.
- **Time** is the execution time of parallelized code.
- **m** and **n** are the length and the width of the matrix.
- **nb** is size of the block.

The size of the matrix and the block can be changed directly from the command line:
```
./plasmatest $name_aplication --dim=mxn --nb=nb
```
Where `dim` is the size of the matrix and `nb` the size of the block.
Additionally, the `--test=n` flag allows you to disable output validity check
which compare the results produced by serial and parallel execution. This might
be useful to reduce execution time for larger matrix.

Use the help flag to know what other parameters can be modified:
```
./plasmatest $name_aplication --help
```

**IMPORTANT**:

1. Remember that the OmpCluster runtime will only be used to
schedule the tasks if `mpirun` is used to execute the application.
```
mpirun -np 3 ./plasmatest $name_aplication --help
```
Not using `mpirun` means using the x86 device offloading that is mostly useful
for debugging OpenMP target code.
2. PBS scripts with more than one node (1+2, 1+4 and 1+8) will not work if the
application is not parallelized using `target nowait`. Be sure to have it
parallelized and functional with the [script](./kahuna_ompcluster_1_node_3_procs.pbs)
using 1 node and 3 processes before trying with multiple nodes.

- **Kahuna**: Several PBS scripts are provided with different configurations.
  Edit the `*.pbs` files to run the application that was attributed to you and
  and eventually modifying some configurations. Docker is not available on
  Kahuna so the scripts are using [Singularity][singularity] to run the
  container.

  + You can test your parallelization on your own computer but you MUST run all
  scalability tests on the Kahuna cluster.

- **Continuous Integration**: You must use the CI server to test your project.
  You must edit the [entrypoint script](./entrypoint.sh) to run your
  application. The application MUST be executed on the CI with the following
  parameters `--dim=256 --nb=64`.

- **Container**: You can execute PLASMA applications with the OmpCluster runtime
  on your own computer using docker and the following command:
```
docker run -v /path/to/plasma/:/root/plasma -it ompcluster/plasma-dev:latest /bin/bash
cd /root/plasma/
```
You can get more information on how to use Docker in the official
[Get Started][docker] guide).

You can also use Singularity (see [here][singularity] for more informations):
```
singularity pull docker://ompcluster/plasma-dev:latest
singularity shell ./plasma-dev_latest.sif
cd /path/to/plasma/
```


Applications
--------------------------------------------------------------------------------
The applications provided by PLASMA are divided into two groups according to
their computational complexity.

- **First Group:** medium computational complexity

| Application     | Description                                                    |
|:-----------:|:----------------------------------------------------------------:|
| GBTRF | Computes an LU factorization of a real m-by-n matrix using partial pivoting with row interchanges. |
| TRMM | Performs a triangular matrix-matrix multiply. |
| GEQRF | Computes a tile QR factorization of a real or complex m-by-n matrix A. |
| HEMM | Performs one of the matrix-matrix operations. |
| HER2K | Performs one of the Hermitian rank 2k operations. |
| LAUUM | Computes the product U∗UH or LH∗L, where the triangular factor U or L is stored in the upper or lower triangular part of the matrix A. |
| PBTRF | Performs the Cholesky factorization of an Hermitian positive matrix A. |
| POTRI | Computes the inverse of a Hermitian positive definite matrix A using the Cholesky factorization. |
| SYMM | Symmetric matrix matrix multiply.|
| SYR2K | Symmetric rank-2k update to a matrix. |

For this group, we strongly suggest to run the scalabiltity tests with the following configuration:
  1. Matrix size of 1024x1024 (`--dim=1024 --nb=256`).
  2. Matrix size of 2048x2048 (`--dim=2048 --nb=512`).
  3. Matrix size of 4096x4096 (`--dim=4096 --nb=1024 --test=n`).
  4. Matrix size of 8192x8192 (`--dim=8192 --nb=2048 --test=n`).

- **Second Group:** large computational complexity

| Application     | Description                                                    |
|:-----------:|:----------------------------------------------------------------:|
| GBSV | Computes the solution to a system of linear equations A x X = B, using the LU factorization.|
| GELS | Solves linear systems involving an m-by-n matrix A using a QRor LQ factorization of A.|
| GEQRS | Computes a minimum-norm solution min || A x X -B || using the QR factorization A = Q x R.|
| GETRI | Computes the inverse of a matrix A using the LUfactorization.|
| HESV | Solves a system of linear equations A x X = B with LTLt factorization.|
| HETRF  | Factorize a Hermitian matrix A using a communication avoiding Aasen's algorithm, followed byband LU factorization.|
| PBSV | Computes the solution to a system of linear equa-tions A x X = B.|
| POSV | Computes the solution to a system of linear equa-tions A x X = B, using Cholesky decomposition.|
| POINV | Performs the Cholesky inversion of a Hermitian Positive definite matrix A. |
| POTRF | Solves a system of linear equations A * X = B using the Cholesky factorization. |

For this group, we strongly suggest to run the scalabiltity tests with the following configuration:
  1. Matrix size of 512x512 (`--dim=512 --nb=128`).
  2. Matrix size of 1024x1024 (`--dim=1024 --nb=256 --test=n`).
  3. Matrix size of 2048x2048 (`--dim=2048 --nb=512 --test=n`).
  4. Matrix size of 4096x4096 (`--dim=4096 --nb=1024 --test=n`).


- **Additional notes**:
  1. Running tests on larger matrices will increase the execution time: do not
  forget that the computing resources of the cluster are shared between all
  users.
  2. The proposed configurations disable output verification for larger matrices
  to reduce the execution time. However, keep running the verification on
  smaller matrices to be sure you parallelization is correct.
  2. You can experiment with various block sizes but keep in mind the overhead
  of the runtime increases when the tasks are too small.  
  3. For more information about PLASMA, read the following [paper][plasma]:
  Dongarra and al. (2019). PLASMA: Parallel linear algebra software for multicore using OpenMP. ACM Transactions on Mathematical Software (TOMS), 45(2), 1-35.


Your tasks
--------------------------------------------------------------------------------
 - **Phase 1**:
   Discovering the project:
   1. Configure your repository to run the application that has been attributed
   to you. You need to update all PBS scripts (which are used to execute on
   Kahuna) and the entrypoint script (use to run validation tests on the CI
   server) with the correct kernel name.
   2. Execute the original version (already parallelized using classical OpenMP
    tasks) of your application on Kahuna to verify that everything is working
   correctly. Familiarize yourself with the parameters of the application. You
   might want to take a first look at the code as well.
 - **Phase 2**:
   Initial parallelization:
   1. Start by profiling the original version using OmpTracing. Look at the
   tracing of the execution and the task-graph to understand better the behavior
   of the application.
   2. Parallelize the application using `target nowait`. Verify that the
   parallel version is still producing correct result **even when using 3 MPI
   processes on a single node**. You MUST WAIT until everything is working
   correctly on a single node before trying to execute on multiple nodes to
   avoid overloading the cluster for your collegues.
 - **Final Phase**:
   Distributed execution and scalability testing:
   1. Start to experiment distributed execution of the application you just
   parallelized on 3 nodes (1 headnode and 2 nodes). If it does not get any
   speedup, look for solutions to improve your parallelization: try to reduce
   the communications between the nodes and experiment with various granularity.
   Keep it mind that you should use multiple levels of parallelization:
   OmpCluster parallelization between the worker nodes and classical OpenMP
   parallelization within eeach working node.
   2. You can try with a higher number of nodes **IF AND ONLY IF you get a
   speedup on 3 nodes**: start with 5 nodes (1 headnode and 4 workers) then with
   9 nodes.

[ompcluster]: https://ompcluster.readthedocs.io/en/latest
[cmake]: https://cmake.org/cmake/help/v3.10/
[cmake-tutorial]: https://cmake.org/cmake/help/latest/guide/tutorial/index.html
[docker]: https://docs.docker.com/get-started/
[singularity]: https://sylabs.io/guides/3.2/user-guide/
[plasma]: https://dl.acm.org/doi/pdf/10.1145/3264491
