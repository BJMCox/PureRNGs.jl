module PureRNGsMooncakeExt

# A draw's base variate depends only on stream bits. Mooncake cannot follow the
# bit casts that decode it, and need not: the distribution transform that
# consumes it carries every parameter derivative.

import PureRNGs
using Mooncake: @zero_derivative, DefaultCtx

const IR = PureRNGs

@zero_derivative DefaultCtx Tuple{typeof(IR._from_bits),Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._open_midpoint),Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._normal_from_bits),Any,Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._exponential_from_bits),Any,Type,Any}
@zero_derivative DefaultCtx Tuple{typeof(IR._exponential_transform),Any,Type,Any}

end
