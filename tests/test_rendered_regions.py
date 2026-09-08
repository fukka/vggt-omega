# Copyright (c) 2026.
"""The grading region of exp_rendered must not leak into the fill.

Region "crop" restricts what is SCORED to the inscribed crop's footprint; the
fill must still see the arm's own hole, or a fill arm under region="crop"
would paint over the whole outside of the footprint and the "fill" would no
longer be the fill being tested.
"""
import json
import os

import numpy as np
import torch

from finetune.eval.exp_rendered import (RenderedWindowDataset, grading_mask_of,
                                        parse_setting)


def _frame(tmp_path, S=32):
    d = tmp_path / "seqA" / "frame_0000"
    d.mkdir(parents=True)
    yy, xx = np.mgrid[:S, :S]
    disc = (np.hypot(xx - S / 2 + .5, yy - S / 2 + .5) <= S * 0.45)      # the "valid" region
    crop = (np.abs(xx - S / 2 + .5) <= S * 0.25) & (np.abs(yy - S / 2 + .5) <= S * 0.25)
    rgb = np.full((S, S, 3), 120, np.uint8)
    rgb_m = rgb.copy(); rgb_m[~disc] = 0
    dep = np.full((S, S), 2.0, np.float32)
    for proj in ("fisheye", "persp"):
        np.save(d / f"{proj}_full_rgb.npy", rgb); np.save(d / f"{proj}_full_depth.npy", dep)
        np.save(d / f"{proj}_masked_rgb.npy", rgb_m); np.save(d / f"{proj}_masked_depth.npy", dep * disc)
        np.save(d / f"mask_{proj}_valid.npy", disc); np.save(d / f"mask_{proj}_in_crop.npy", crop)
    np.save(d / "persp_crop_rgb.npy", rgb); np.save(d / "persp_crop_depth.npy", dep)
    np.save(d / "mask_persp_crop_valid.npy", np.ones((S, S), bool))
    json.dump({}, open(d / "meta.json", "w"))
    return disc, crop


def test_parse_crop_setting():
    assert parse_setting("persp_crop") == ("persp", "crop")
    assert grading_mask_of("persp_crop") == "mask_persp_crop_valid"
    assert grading_mask_of("persp_masked") == "mask_persp_valid"


def test_region_crop_restricts_grading_but_not_fill(tmp_path):
    disc, crop = _frame(tmp_path)
    own = RenderedWindowDataset(str(tmp_path), "persp_fill_mean", 1)[0]
    com = RenderedWindowDataset(str(tmp_path), "persp_fill_mean", 1, region="crop")[0]
    # Scored pixels: own = disc, crop = disc & footprint.
    assert np.array_equal(own["valid_masks"][0].numpy(), disc)
    assert np.array_equal(com["valid_masks"][0].numpy(), disc & crop)
    # The image the model sees is identical: the fill used the own hole both times.
    assert torch.equal(own["images"], com["images"])
    img = own["images"][0].permute(1, 2, 0).numpy()
    assert (img.max(-1) > 0).all(), "fill left black pixels"


def test_crop_arm_is_graded_on_its_whole_frame_in_both_regions(tmp_path):
    _frame(tmp_path)
    for region in ("own", "crop"):
        s = RenderedWindowDataset(str(tmp_path), "persp_crop", 1, region=region)[0]
        assert s["valid_masks"][0].all()
