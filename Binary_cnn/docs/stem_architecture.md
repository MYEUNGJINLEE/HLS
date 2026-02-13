# Stem Processor Architecture (Current Code)

## 1) High-Level Block View

```mermaid
flowchart LR
  DRAM_RGB[DRAM Model\nRGB Input] --> LB0
  WREQ[Weight Request] --> DRAM_W[DRAM Model\nWeight Stream]
  DRAM_W --> WLOAD
  STAT[Status Stream] --> HOST[Host/TB Scheduler]

  subgraph STEM[StemProcessor::run]
    WLOAD[Weight/Param Load\n(w0~w3, shift0~3, bias0~3)]
    LB0[line_buf_a\n(RGB -> Conv0 window)]

    C0[Conv0 3x3 s2\n3 -> 32]

    C1[Conv1 1x1\n32 -> 16]
    LB1[conv1_buf\n(Conv2 input line buffer)]
    C2[Conv2 3x3 s2\n16 -> 32]

    LBM[mp_buf\n(MaxPool input line buffer)]
    MP[MaxPool 2x2 s2\n32 -> 32]

    CAT[Row Staging + Concat\n(ping-pong row tiles)]
    C3[Conv3 1x1\n64 -> 32]
    PACK[Pack int8 x32]

    LB0 --> C0
    C0 --> C1 --> LB1 --> C2 --> CAT
    C0 --> LBM --> MP --> CAT
    CAT --> C3 --> PACK

    WLOAD --> C0
    WLOAD --> C1
    WLOAD --> C2
    WLOAD --> C3

    CTRL[Control FSM\nPRELOAD/PRIME/RUN/DONE]
    CTRL --> WLOAD
    CTRL --> CAT
    CTRL --> PACK
  end

  PACK --> OUT[Output Stream\n(outbuffer 역할)]
  STEM --> WREQ
  STEM --> STAT
```

## 2) What Differs from the Simple Linear Diagram

Your sketch was:

`DRAM -> linebuffer -> stem -> outbuffer`

Current implementation is:

`DRAM(RGB + weights) -> preload -> [line_buf_a + Conv0] -> branch(Path-A Conv1/Conv2, Path-B MaxPool) -> ping-pong concat -> Conv3 -> output_stream`

Control plane is explicit:

`weight_req/status_stream` are used for deterministic preload handshake and tile-level progress.

So it is not a single linear "linebuffer -> one stem block" pipeline; it is a branched stem graph inside one top function.

## 3) Code Anchor Points

- Top interface: `Binary_cnn/src/stem/stem_processor.h`
- Internal buffers: `Binary_cnn/src/stem/stem_processor.h`
- Main schedule and branch/concat flow: `Binary_cnn/src/stem/stem_processor.cpp`
