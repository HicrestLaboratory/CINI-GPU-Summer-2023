# CINI-GPU-Summer-2023
GPU Programming: Hands on session. CINI HPC Summer School 2023

## Prefix Sum

The code could be launched with the following command line:

```
PrefixSum n v
```

The parameter _n_ define the problem size: the vector on which the prefix sum is performed has dimension 2^n.

If you wanna print all the involved vectors, you can set the verbose parameter _'v'_ as 1, otherwise, it must be 0.

## Matrix multiplication

The code could be launched with the following command line:

```
MatrixGemm_ex e v [CPU_flag = 1]
```

The parameter _e_ defines the problem size: the total number of elements in the matrix is 2^e, this also means that _'e'_ must be even. Moreover, due to size reasons, _'e'_ must also be greater-equal than 12.

If you wanna print all the involved matrices, you can turn it off by putting the optional parameter _'CPU_flag'_ as 0.

Since the CPU computation for dimensions greater than 15 starts to become time expensive, this could be turned off by putting the optional parameter _'CPU_flag'_ as 0.
