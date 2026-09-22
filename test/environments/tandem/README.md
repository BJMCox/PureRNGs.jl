# Tandem conformance

Use Julia 1.13 with sibling `PureRNGs.jl` and `TandemRNG.jl` checkouts. Both
checkouts must include the Tandem bridge and MLDataDevices binding changes.
The relative paths in `Project.toml` select those sources.

From the PureRNGs root, run:

```sh
julia --project=test/environments/tandem -e 'using Pkg; Pkg.instantiate(); Pkg.test()'
```

The default suite checks the CPU bridge and Reactant on its CPU backend. Set
`TANDEM_TEST_CUDA=true` to add CUDA residence and device-kernel checks. Set
`TANDEM_REACTANT_BACKEND=gpu` to run the Reactant checks on a supported GPU.
CUDA and Reactant must use compatible devices in that combined run.
