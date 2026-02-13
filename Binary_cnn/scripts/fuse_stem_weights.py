#!/usr/bin/env python3
import argparse
import math


def iter_tokens(path):
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            for tok in line.split():
                if tok.startswith("#"):
                    break
                yield tok


def read_weight_bits(path):
    bits = []
    for tok in iter_tokens(path):
        if tok in ("0", "1"):
            bits.append(int(tok))
        elif tok == "+1":
            bits.append(0)
        elif tok == "-1":
            bits.append(1)
        else:
            raise ValueError(f"Invalid weight token: {tok}")
    return bits


def read_float_list(path, count=None):
    vals = []
    for tok in iter_tokens(path):
        vals.append(float(tok))
        if count is not None and len(vals) >= count:
            break
    if count is not None and len(vals) < count:
        raise ValueError(f"Expected {count} floats in {path}, got {len(vals)}")
    return vals


def clamp(val, lo, hi):
    return max(lo, min(hi, val))


def scale_to_shift(scale, shift_min=-6, shift_max=6):
    if scale == 0.0:
        return 0
    sh = int(round(math.log(abs(scale), 2)))
    return clamp(sh, shift_min, shift_max)


def pack_bits(bits):
    u = 0
    for i, b in enumerate(bits):
        if b:
            u |= (1 << i)
    return u


def pack_param(shift, bias):
    sh = shift & 0xFF
    bi = bias & 0xFFFF
    return sh | (bi << 8)


def main():
    ap = argparse.ArgumentParser(description="Fuse stem weights + BN into packed stream.")
    ap.add_argument("--weight", required=True, help="Weight txt (0/1 or +1/-1 tokens)")
    ap.add_argument("--bn-scale", required=True, help="BN scale txt (float)")
    ap.add_argument("--bn-bias", required=True, help="BN bias txt (float)")
    ap.add_argument("--out", required=True, help="Output packed stream file")
    ap.add_argument("--shift-min", type=int, default=-6)
    ap.add_argument("--shift-max", type=int, default=6)
    args = ap.parse_args()

    # Layer sizes
    CONV0_OUT_CH = 32
    CONV0_IN_CH = 3
    CONV1_OUT_CH = 16
    CONV1_IN_CH = 32
    CONV2_OUT_CH = 32
    CONV2_IN_CH = 16
    CONV3_OUT_CH = 32
    CONV3_IN_CH = 64

    total_bn = CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH + CONV3_OUT_CH

    weights = read_weight_bits(args.weight)
    bn_scale = read_float_list(args.bn_scale, total_bn)
    bn_bias = read_float_list(args.bn_bias, total_bn)

    # Slice BN per layer
    bn0_s = bn_scale[0:CONV0_OUT_CH]
    bn0_b = bn_bias[0:CONV0_OUT_CH]
    bn1_s = bn_scale[CONV0_OUT_CH:CONV0_OUT_CH + CONV1_OUT_CH]
    bn1_b = bn_bias[CONV0_OUT_CH:CONV0_OUT_CH + CONV1_OUT_CH]
    bn2_s = bn_scale[CONV0_OUT_CH + CONV1_OUT_CH:CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH]
    bn2_b = bn_bias[CONV0_OUT_CH + CONV1_OUT_CH:CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH]
    bn3_s = bn_scale[CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH:]
    bn3_b = bn_bias[CONV0_OUT_CH + CONV1_OUT_CH + CONV2_OUT_CH:]

    # Helper to flip weights per output channel if scale < 0
    def maybe_flip(bits, scale):
        if scale >= 0:
            return bits
        return [1 - b for b in bits]

    out_lines = []
    idx = 0

    # Conv0 weights
    for oc in range(CONV0_OUT_CH):
        for ic in range(CONV0_IN_CH):
            bits = weights[idx:idx + 9]
            if len(bits) < 9:
                raise ValueError("Not enough weight bits for Conv0")
            bits = maybe_flip(bits, bn0_s[oc])
            out_lines.append(pack_bits(bits))
            idx += 9
    # Conv0 params
    for oc in range(CONV0_OUT_CH):
        shift = scale_to_shift(bn0_s[oc], args.shift_min, args.shift_max)
        bias = int(round(bn0_b[oc]))
        out_lines.append(pack_param(shift, bias))

    # Conv1 weights
    for oc in range(CONV1_OUT_CH):
        bits = weights[idx:idx + CONV1_IN_CH]
        if len(bits) < CONV1_IN_CH:
            raise ValueError("Not enough weight bits for Conv1")
        bits = maybe_flip(bits, bn1_s[oc])
        out_lines.append(pack_bits(bits))
        idx += CONV1_IN_CH
    # Conv1 params
    for oc in range(CONV1_OUT_CH):
        shift = scale_to_shift(bn1_s[oc], args.shift_min, args.shift_max)
        bias = int(round(bn1_b[oc]))
        out_lines.append(pack_param(shift, bias))

    # Conv2 weights
    for oc in range(CONV2_OUT_CH):
        for ic in range(CONV2_IN_CH):
            bits = weights[idx:idx + 9]
            if len(bits) < 9:
                raise ValueError("Not enough weight bits for Conv2")
            bits = maybe_flip(bits, bn2_s[oc])
            out_lines.append(pack_bits(bits))
            idx += 9
    # Conv2 params
    for oc in range(CONV2_OUT_CH):
        shift = scale_to_shift(bn2_s[oc], args.shift_min, args.shift_max)
        bias = int(round(bn2_b[oc]))
        out_lines.append(pack_param(shift, bias))

    # Conv3 weights
    for oc in range(CONV3_OUT_CH):
        bits = weights[idx:idx + CONV3_IN_CH]
        if len(bits) < CONV3_IN_CH:
            raise ValueError("Not enough weight bits for Conv3")
        bits = maybe_flip(bits, bn3_s[oc])
        out_lines.append(pack_bits(bits))
        idx += CONV3_IN_CH
    # Conv3 params
    for oc in range(CONV3_OUT_CH):
        shift = scale_to_shift(bn3_s[oc], args.shift_min, args.shift_max)
        bias = int(round(bn3_b[oc]))
        out_lines.append(pack_param(shift, bias))

    with open(args.out, "w") as f:
        for u in out_lines:
            f.write(f"0x{u:016X}\n")

    print(f"Wrote {len(out_lines)} packed words to {args.out}")


if __name__ == "__main__":
    main()
