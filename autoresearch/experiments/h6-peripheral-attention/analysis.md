# H6 analysis

## CPU pilot (2026-08-26, EXPLORATORY — and it did NOT confirm the direction)

Same reduced setting as the H5 pilot (seq131 even/odd frame split, 252 px,
10 epochs). Result: near rim +2.1% (the target zone, unimproved), near
center −22.8%, center −4.0%, far +2.8%.

Two honest reads, both consequential:

1. **The pilot violates the module's premise.** "Adjacent" pilot frames are
   100 capture-frames (~3.3 s) apart — enormous viewpoint change; cross-frame
   attention has little usable correspondence to route. This is a weak test,
   not a refutation. BUT the same trap was about to hit #36 on the box: the
   trainer subsampled frames UNIFORMLY to --max-frames, recreating sparse
   pairs on full sequences. Fixed: Seq gains dense=True (contiguous middle
   block, video-rate neighbors); the H6 trainer and eval now use it (H5
   keeps uniform sampling — its mv term is auxiliary; scene diversity
   matters more there).
2. **"Center untouched" is token-level, not output-level.** The DPT head
   mixes tokens spatially, so rim-token updates CAN move center outputs
   (here they helped: −22.8% near center; elsewhere they could hurt).
   Protocol wording corrected: structural safety claims for H6 cover the
   POSE path (reads original feats) and rim-query locality at the token
   level; center-depth safety is empirical, same as H5.

The GPU run (#36, dense windows) is the real test; the pilot's job was to
catch exactly this kind of setup error before burning box time — it did.
