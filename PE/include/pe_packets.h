#ifndef PE_PACKETS_H
#define PE_PACKETS_H

#include "pe_types.h"

enum pe_pkt_kind_t {
    PE_PKT_KIND_WEIGHT = 0,
    PE_PKT_KIND_BN = 1,
    PE_PKT_KIND_END = 2
};

enum pe_pkt_op_t {
    PE_PKT_OP_CONV1X1 = 0,
    PE_PKT_OP_CONV3X3 = 1,
    PE_PKT_OP_DW3X3 = 2
};

struct PEPacketHeader {
    int kind;
    int op;
    int oc_idx;
    int ic_group;
    int aux;
};

inline int pe_pkt_op_from_cfg(pe_op_t op) {
    if (op == PE_OP_CONV1X1) {
        return PE_PKT_OP_CONV1X1;
    }
    if (op == PE_OP_CONV3X3) {
        return PE_PKT_OP_CONV3X3;
    }
    if (op == PE_OP_DW3X3) {
        return PE_PKT_OP_DW3X3;
    }
    return -1;
}

inline PEPacketHeader pe_pkt_header(const pe_weight_pkt_t &pkt) {
    PEPacketHeader h;
    h.kind = (int)pkt.slc<4>(124);
    h.op = (int)pkt.slc<4>(120);
    h.oc_idx = (int)pkt.slc<8>(112);
    h.ic_group = (int)pkt.slc<8>(104);
    h.aux = (int)pkt.slc<8>(96);
    return h;
}

inline ac_int<96, false> pe_pkt_payload(const pe_weight_pkt_t &pkt) {
    return pkt.slc<96>(0);
}

inline pe_weight_pkt_t pe_make_packet(
    int kind,
    int op,
    int oc_idx,
    int ic_group,
    int aux,
    const ac_int<96, false> &payload
) {
    pe_weight_pkt_t pkt = 0;
    pkt.set_slc(124, (ac_int<4, false>)(kind & 0xF));
    pkt.set_slc(120, (ac_int<4, false>)(op & 0xF));
    pkt.set_slc(112, (ac_int<8, false>)(oc_idx & 0xFF));
    pkt.set_slc(104, (ac_int<8, false>)(ic_group & 0xFF));
    pkt.set_slc(96, (ac_int<8, false>)(aux & 0xFF));
    pkt.set_slc(0, payload);
    return pkt;
}

inline pe_weight_pkt_t pe_make_weight_packet(
    int op,
    int oc_idx,
    int ic_group,
    const ac_int<96, false> &payload
) {
    return pe_make_packet(PE_PKT_KIND_WEIGHT, op, oc_idx, ic_group, 0, payload);
}

inline pe_weight_pkt_t pe_make_bn_packet(int op, int oc_idx, pe_shift_t shift, pe_bias_t bias) {
    ac_int<96, false> payload = 0;
    payload.set_slc(0, shift.slc<8>(0));
    payload.set_slc(8, bias.slc<16>(0));
    return pe_make_packet(PE_PKT_KIND_BN, op, oc_idx, 0, 0, payload);
}

inline pe_weight_pkt_t pe_make_end_packet(void) {
    ac_int<96, false> payload = 0;
    return pe_make_packet(PE_PKT_KIND_END, 0, 0, 0, 0, payload);
}

inline ac_int<64, false> pe_payload_conv1(const pe_weight_pkt_t &pkt) {
    return pe_pkt_payload(pkt).slc<64>(0);
}

inline ac_int<72, false> pe_payload_conv3(const pe_weight_pkt_t &pkt) {
    return pe_pkt_payload(pkt).slc<72>(0);
}

inline ac_int<9, false> pe_payload_dw3(const pe_weight_pkt_t &pkt) {
    return pe_pkt_payload(pkt).slc<9>(0);
}

inline pe_shift_t pe_payload_bn_shift(const pe_weight_pkt_t &pkt) {
    pe_shift_t v = 0;
    v.set_slc(0, pe_pkt_payload(pkt).slc<8>(0));
    return v;
}

inline pe_bias_t pe_payload_bn_bias(const pe_weight_pkt_t &pkt) {
    pe_bias_t v = 0;
    v.set_slc(0, pe_pkt_payload(pkt).slc<16>(8));
    return v;
}

#endif // PE_PACKETS_H
