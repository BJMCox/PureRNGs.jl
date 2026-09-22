# Acknowledgements

## Julia prior art

To our knowledge, adapting the explicit-state RNG API pattern common in functional programming to Julia was first explored in ImmutableRNGs.jl.[^immutable]
Here, a draw returns both a sample and the next state without mutating the supplied state.
We acknowledge Uwe Hernandez Acosta's work as prior art for this aspect of PureRNGs' API.

Random123.jl[^random123jl] and PhiloxRNG.jl[^philoxrng] provided Julia implementation and performance references during development.
Random123.jl exposes counter-based generators through Julia's `Random` interface.
PhiloxRNG.jl exposes stateless Philox functions for CPU and GPU use.

RNGTest.jl[^rngtest] provides the Julia interface used for our TestU01 and BigCrush checks.
We thank these packages' authors and contributors for their work.

## Algorithms and statistical tests

Philox and Threefry come from the work of Salmon, Moraes, Dror, and Shaw[^random123paper] and the Random123 reference implementation.[^random123]
Our core tests use Random123's known-answer vectors.
ChaCha follows Daniel J. Bernstein's design.[^chacha]

CPU normal sampling uses Michael J. Wichura's AS241 approximation.[^wichura]
Native GPU normal sampling uses Mike Giles' inverse-error-function approximation.[^giles]
These approximation methods and their published coefficients are prior work, not new algorithms introduced by PureRNGs.

Our statistical validation uses TestU01, by Pierre L'Ecuyer and Richard Simard,[^testu01] and PractRand.[^practrand]
The [Reproducibility](@ref) page describes the scope of that evidence.

## References

Software citations identify the versions consulted, not runtime dependencies or guarantees of API compatibility.
Software repositories were accessed on 22 September 2026.

[^immutable]: Hernandez Acosta, Uwe. *ImmutableRNGs.jl* (version 0.1.0) [Computer software]. GitHub. [Source repository](https://github.com/szabo137/ImmutableRNGs.jl).

[^random123jl]: Sunoru and contributors. *Random123.jl* (version 1.7.1) [Computer software]. JuliaRandom. [Source repository](https://github.com/JuliaRandom/Random123.jl).

[^philoxrng]: Zimmerberg, Nathan, and Patrick Kelly. *PhiloxRNG.jl* (version 1.1.2) [Computer software]. medyan-dev. [Source repository](https://github.com/medyan-dev/PhiloxRNG.jl).

[^rngtest]: Noack, Andreas, and contributors. *RNGTest.jl* (version 1.6.1) [Computer software]. JuliaRandom. [Source repository](https://github.com/JuliaRandom/RNGTest.jl).

[^random123paper]: Salmon, John K., Mark A. Moraes, Ron O. Dror, and David E. Shaw. (2011). *Parallel Random Numbers: As Easy as 1, 2, 3*. In *Proceedings of the International Conference for High Performance Computing, Networking, Storage and Analysis (SC '11)*, article 16, pp. 1–12. ACM. [doi:10.1145/2063384.2063405](https://doi.org/10.1145/2063384.2063405). [Author-hosted paper](https://www.thesalmons.org/john/random123/papers/random123sc11.pdf).

[^random123]: D. E. Shaw Research. *Random123* (version 1.14.0) [Computer software]. [Source repository and reference vectors](https://github.com/DEShawResearch/random123/tree/v1.14.0).

[^chacha]: Bernstein, Daniel J. (2008). *ChaCha, a variant of Salsa20*. In *Workshop Record of SASC 2008: The State of the Art of Stream Ciphers*. [Author's publication record and paper](https://cr.yp.to/papers.html#chacha).

[^wichura]: Wichura, Michael J. (1988). *Algorithm AS 241: The Percentage Points of the Normal Distribution*. *Journal of the Royal Statistical Society. Series C (Applied Statistics)*, **37**(3), 477–484. [doi:10.2307/2347330](https://doi.org/10.2307/2347330).

[^giles]: Giles, Mike. (2012). *Approximating the erfinv Function*. In *GPU Computing Gems Jade Edition*, chapter 10, pp. 109–116. Morgan Kaufmann. [doi:10.1016/B978-0-12-385963-1.00010-1](https://doi.org/10.1016/B978-0-12-385963-1.00010-1). [Author's paper and coefficient-generation code](https://people.maths.ox.ac.uk/gilesm/codes/erfinv/).

[^testu01]: L'Ecuyer, Pierre, and Richard Simard. (2007). *TestU01: A C Library for Empirical Testing of Random Number Generators*. *ACM Transactions on Mathematical Software*, **33**(4), article 22, pp. 1–40. [doi:10.1145/1268776.1268777](https://doi.org/10.1145/1268776.1268777).

[^practrand]: Doty-Humphrey, Chris. *PractRand (Practically Random)* (version 0.96) [Computer software]. [Project website and documentation](https://pracrand.sourceforge.net/).
