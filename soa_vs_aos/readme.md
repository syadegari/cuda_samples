# AoS versus SoA: remove wasted memory transactions

Each record has twenty floats (80 bytes). Each thread adds 10 to `u` and `y[0]`; the other eighteen fields are untouched.
The two kernels perform the same arithmetic on identical logical input values. Only the physical layout changes.
