---
name: Bug report
about: Report incorrect results, an unexpected error, or a broken backend path
title: ""
labels: bug
---

## What happened

Describe the observed behavior and the behavior you expected.

## Environment

- Julia version (`versioninfo()`):
- PureRNGs commit:
- Backend (CPU, CUDA, AMDGPU, Metal, Reactant) and its package version:
- Thread count (`Threads.nthreads()`):

## Minimal script

A script that runs from a fresh session and shows the problem.

```julia
using PureRNGs

rng = Philox4x32(123456)
```

## Output

The full error message and stack trace, or the wrong values with the values you expected.
