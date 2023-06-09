# Matrix multiplication

The code could be launched with the following command line:

```
MatrixGemm_ex e v [CPU_flag = 1]
```

The parameter _e_ defines the problem size: the total number of elements in the matrix is 2^e, this also means that _'e'_ must be even. Moreover, due to size reasons, _'e'_ must also be greater-equal than 12.

If you wanna print all the involved matrices, you can turn it off by putting the optional parameter _'CPU_flag'_ as 0.

Since the CPU computation for dimensions greater than 15 starts to become time expensive, this could be turned off by putting the optional parameter _'CPU_flag'_ as 0.
