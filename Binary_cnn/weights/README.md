# Stem Weight Files

Place real stem parameter files in this folder for `make stem-tb-weight`.

Required files:

- `stem_weights.txt`
- `stem_bn_scale.txt`
- `stem_bn_bias.txt`

## `stem_weights.txt` format

- Whitespace-separated tokens
- Allowed tokens per weight: `0`, `1`, `+1`, `-1`
- `#` starts a comment line
- Total tokens: `8032`

Token order:

1. Conv0: `32 * 3 * 9`
2. Conv1: `16 * 32`
3. Conv2: `32 * 16 * 9`
4. Conv3: `32 * 64`

## BN file format

- `stem_bn_scale.txt`: 112 float tokens
- `stem_bn_bias.txt`: 112 float tokens
- `#` starts a comment line

BN token order:

1. Conv0: 32
2. Conv1: 16
3. Conv2: 32
4. Conv3: 32

## Commands

- Auto mode: `make stem-tb` (all 3 files -> weight mode, none -> no-weight mode)
- Real files: `make stem-tb-weight`
- No files (fallback mode): `make stem-tb-no-weight`
