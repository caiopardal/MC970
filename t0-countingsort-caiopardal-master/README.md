OpenMP Parallel For: Counting Sort
================================================================================

Counting Sort is an integer sorting algorithm. For each element in the array,
this algorithm will count the number of elements in the array which are
smaller than the current one. This way, it is possible to know which position
the current element should be stored in the sorted array. You can read more
about the algorithm at [Wikipedia][wiki].

In this assignment you will have to parallelize an implementation of the
Counting Sort algorithm using OpenMP `parallel for` construct.

[wiki]: https://en.wikipedia.org/wiki/Counting_sort

### Input

The program expects three lines as input:

- The first line must contain a single integer which represents the number of
  threads OpenMP should use;
- The second line must contain a single integer which represents the number of
  elements in the array;
- The third and last line must have one or more floats which are the array
  elements.

For example:

```
2
5
5918.97 5213.70 1442.98 4845.91 2492.30
```

### Output

The program outputs two lines to stdout: the first is the ordered vector and the
second is the runtime in seconds.

Tasks & Rules
--------------------------------------------------------------------------------

You should do the following tasks:

- [ ] Understand the serial code in `src/countsort-serial.c`
- [ ] Parallelize the code using OpenMP in the file `src/countsort-parallel.c`
- [ ] Run both versions and compare them. Did you get any speedup?

You must **not** change the serial implementation, only the parallel one.

Grading
--------------------------------------------------------------------------------

Your assignment will be evaluated in terms of:

- Correctness: your program returns the correct result;
- Performance: your program runs faster than the serial implementation.

In order to test your solution, you can use one of the inputs available inside
the `tests/` directory. Whenever you push your changes to GitHub, the Continuous
Integration (CI) system will compile, run and execute your program using those
tests. Your grade will **not** be computed from the CI runs because other
processes may be running in the server, thus interfering with runtime and
speedup.

Your grade will be computed using an automated routine restricted to the
instructors and TAs. This routine will be run after the assignment deadline,
using the latest commit push to the branch `master` before the deadline. Your
code will be ensured to run in an environment with no competition for resources.

**Note:** Both the CI and automatic grading routine expect your the output of
your program to be formatted correctly. For that reason, you should not add
`printf`s or any other function that writes to `stdout`, otherwise your
assignment will be considered incorrect.

**Note:** Tampering with the serial implementation or the tests is considered
cheating and will result in disqualification of the assignment.

Compiling & Running
--------------------------------------------------------------------------------

After you have accepted this assignment on the course's GitHub Classroom page,
clone it to your machine. First you have to generate the build system using
[CMake](https://cmake.org/). Make sure you have it installed! You will also need
an OpenMP compatible compiler. If you use Linux, you are good to go. For MacOS
users, you can install the OpenMP libraries via [Homebrew](https://brew.sh/)
with the following command:

```bash
# Only for MacOS users
brew install libomp
```

Then, run the following commands:

```bash
# Where the build will live
mkdir build && cd build

# Generate the Makefiles
cmake -DCMAKE_BUILD_TYPE=Release ..
```

Having done that, still inside the `build` directory, run `make` to compile
everything. Finally, from the root directory of the repo you can execute the
serial and parallel versions like so:

```bash
build/serial tests/1.in
build/parallel tests/2.in
```

If you have any doubts or run into problems, please contact the TAs. Happy
coding! :smile: :keyboard:

Contribute
--------------------------------------------------------------------------------

Found a typo? Something is missing or broken? Have ideas for improvement? The
instructor and the TAs would love to hear from you!

About
--------------------------------------------------------------------------------

This repository is one of the assignments handed out to the students in course
"MO644/MC970 - Introduction to Parallel Programming" offered by the Institute of
Computing at Unicamp.
