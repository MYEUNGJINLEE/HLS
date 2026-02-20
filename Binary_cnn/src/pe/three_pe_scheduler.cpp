#include "three_pe_scheduler.h"

void LayerMemReader::run(
    stem_packed_act_t              src_buf[SCHED_MAX_PIXELS],
    int                            pixels,
    ac_channel<stem_packed_act_t> &out_stream
) {
    READ_PIXELS:
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < pixels; i++) {
        out_stream.write(src_buf[i]);
    }
}

void LayerMemWriter::run(
    ac_channel<stem_packed_act_t> &in_stream,
    stem_packed_act_t              dst_buf[SCHED_MAX_PIXELS],
    int                            pixels,
    ac_channel<stem_packed_act_t> &forward_stream,
    bool                           enable_forward
) {
    WRITE_PIXELS:
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < pixels; i++) {
        stem_packed_act_t pkt = in_stream.read();
        dst_buf[i] = pkt;
        if (enable_forward) {
            forward_stream.write(pkt);
        }
    }
}

int ThreePEScheduler::input_pixels_for(const ThreePECfg &cfg) {
    return cfg.pe_a.in_h * cfg.pe_a.in_w;
}

int ThreePEScheduler::output_pixels_for(const ThreePECfg &cfg) {
    if (cfg.topo == TOPO_BRANCH_CAT) {
        return cfg.cat_conv.out_h * cfg.cat_conv.out_w;
    }
    if (cfg.topo == TOPO_SHORTCUT) {
        return cfg.pe_b.out_h * cfg.pe_b.out_w;
    }
    return cfg.pe_a.out_h * cfg.pe_a.out_w;
}

void ThreePEScheduler::run(
    const SchedulerCfg            &cfg,
    ac_channel<stem_packed_act_t> &input_stream,
    ac_channel<stem_packed_bw_t>  &weight_stream,
    ac_channel<stem_packed_act_t> &output_stream
) {
    const int num_layers = cfg.num_layers;
    if (num_layers <= 0 || num_layers > SCHED_MAX_LAYERS) {
        return;
    }

    // Bootstrap: first layer input from external stream -> ping buffer.
    const int first_in_pixels = input_pixels_for(cfg.layers[0]);
    if (first_in_pixels <= 0 || first_in_pixels > SCHED_MAX_PIXELS) {
        return;
    }

    LOAD_FIRST_INPUT:
    #pragma hls_pipeline_init_interval 1
    for (int i = 0; i < first_in_pixels; i++) {
        feat_ping[i] = input_stream.read();
    }

    bool ping_is_src = true;
    int expected_in_pixels = first_in_pixels;

    // Layer-by-layer schedule:
    // Stage-1 reader -> Stage-2 3PE block -> Stage-3 writer(forward on last layer)
    SCHEDULE_LAYERS:
    for (int li = 0; li < num_layers; li++) {
        const ThreePECfg &layer_cfg = cfg.layers[li];
        const int in_pixels = input_pixels_for(layer_cfg);
        const int out_pixels = output_pixels_for(layer_cfg);

        if (in_pixels <= 0 || in_pixels > SCHED_MAX_PIXELS) {
            return;
        }
        if (out_pixels <= 0 || out_pixels > SCHED_MAX_PIXELS) {
            return;
        }
        if (li > 0 && in_pixels != expected_in_pixels) {
            // Scheduler requires strict layer-to-layer pixel count continuity.
            return;
        }

        stem_packed_act_t *src_buf = ping_is_src ? feat_ping : feat_pong;
        stem_packed_act_t *dst_buf = ping_is_src ? feat_pong : feat_ping;

        ac_channel<stem_packed_act_t> layer_in_stream;
        ac_channel<stem_packed_act_t> layer_out_stream;

        reader.run(src_buf, in_pixels, layer_in_stream);
        pe_block.run(layer_cfg, layer_in_stream, weight_stream, layer_out_stream);
        writer.run(layer_out_stream, dst_buf, out_pixels, output_stream, (li == (num_layers - 1)));

        ping_is_src = !ping_is_src;
        expected_in_pixels = out_pixels;
    }
}
