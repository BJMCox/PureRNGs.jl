const FAMILY_BITS = UInt32(0x00000000)

@inline _block(rng::Philox2x32, family::UInt32, block::UInt64) = _philox2x32(
    (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32)),
    rng.key,
)

@inline _block(rng::Threefry2x32, family::UInt32, block::UInt64) = _threefry2x32(
    (block % UInt32, (family << 24) | (((block >> 32) & 0x00ffffff) % UInt32)),
    rng.key,
)

@inline _block(rng::Philox4x32, family::UInt32, block::UInt64) =
    _philox4x32((block % UInt32, (block >> 32) % UInt32, family, UInt32(0)), rng.key)

@inline _block(rng::Threefry4x32, family::UInt32, block::UInt64) =
    _threefry4x32((block % UInt32, (block >> 32) % UInt32, family, UInt32(0)), rng.key)

@inline _block(rng::Philox2x64, family::UInt32, block::UInt64) =
    _philox2x64((block, UInt64(family)), rng.key)

@inline _block(rng::Threefry2x64, family::UInt32, block::UInt64) =
    _threefry2x64((block, UInt64(family)), rng.key)

@inline _block(rng::Philox4x64, family::UInt32, block_lo::UInt64, block_hi::UInt64) =
    _philox4x64((block_lo, block_hi, UInt64(family), UInt64(0)), rng.key)

@inline _block(rng::Threefry4x64, family::UInt32, block_lo::UInt64, block_hi::UInt64) =
    _threefry4x64((block_lo, block_hi, UInt64(family), UInt64(0)), rng.key)
