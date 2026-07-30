# `array_lambda_cuda`

A simple example of doing a CUDA launch over elements of a 2D array,
mimicking how AMReX's launches work.

This is meant for pedagogy.

This is written in C++20.

## Compiling

Compile as:

```
nvcc -x cu -std=c++20 --extended-lambda array_lambda.cpp -o array_lambda
```

> [!NOTE]
> Depending on the CUDA version, you may need a different (older) version of GCC.  This
> can be done via `-ccbin=g++-14`

