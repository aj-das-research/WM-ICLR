# External IWS baseline: exact blocker and smallest fair next step

Audit date: 19 September 2026. This audit changes no experiment, checkpoint, reserved-data access policy, or paper claim. No GPU jobs were submitted. Only public repository/configuration information and small checkpoint metadata ranges were downloaded; no checkpoint tensor payloads or dataset payloads were read.

**Conclusion: the released RLA-WM IWS checkpoints cannot reconstruct the DINOv3-L image encoder offline.** Their `encoder` is the *residual latent-action encoder*, a different network. Approved access to the exact DINOv3-L weights, or an already legitimately acquired local copy of those exact weights, remains necessary. The public RLA-WM release does not supply that copy. Moreover, obtaining it enables baseline reproduction; it does not by itself make our current DINOv2 feature-MSE results comparable to the upstream DINOv3/RGB metrics.

## Pinned evidence

- Local upstream checkout is clean at `6f19048758699bf9a152eaed5ac6dbf1caa07c18`. The [official setup instructions](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/docs/setup.md) explicitly require granted DINOv3 access. [Checkpoint documentation](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/docs/data-and-checkpoints.md) distinguishes RLA encoders, feature-to-image decoders, and flow models.
- Public [RLA-WM model release](https://huggingface.co/xyzhang368/RLA-WM/tree/781412392f8a5da8c797d7c3bbb054ae3b5d9072) revision: `781412392f8a5da8c797d7c3bbb054ae3b5d9072`. Its API manifest contains 1,947 files, `gated=false`, model-card license `cc-by-4.0`, and no DINOv3-named weight asset. This filename observation alone is not the basis for the conclusion; checkpoint metadata and serialization code were also checked below.
- Exact [Meta backbone](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m): `facebook/dinov3-vitl16-pretrain-lvd1689m`, revision `ea8dc2863c51be0a264bab82070e3e8836b02d51`. Its public API reports `gated=manual`; an unauthenticated HEAD request for that revision's `model.safetensors` returned HTTP 401. The model card displays a separate DINOv3 license and an access-agreement requirement. This probe did not inspect local credentials or establish whether a user's account has approval.
- Small RLA-WM release configs were read at the pinned model revision. PushT RLA config SHA256: `ae646935831c75ba9f666ebd6d9fd90a494aab13c6a600e5b83f9e2a26208dc7`; PushT flow config: `b125f2a3741f8dca961464514fbaec4c2af84f5824c92c69bf70a9512df114f1`. Both specify 1,024 DINO channels. The RLA encoder is `SimpleTokenTransformer`, not a vision-transformer backbone.

## Why the apparent encoder shortcut does not work

The [trainer](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/src/trainers/rla_wm_trainer.py#L116-L145) constructs `self.dino_extractor` separately, then loads `self.models['encoder']` and `self.models['decoder']` from RLA checkpoints. The former maps RGB to DINOv3 patch features. The latter pair encodes feature residuals into 32 latent-action tokens and reconstructs feature changes.

The [checkpoint writer](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/src/trainers/basic.py#L240-L264) serializes only non-frozen entries in `self.models`. The separate extractor is never such an entry. RLA-WM training freezes its RLA encoder, RLA decoder, and image decoder; consequently its released flow checkpoint contains the flow model, not a complete image-to-future-image pipeline.

The [DINO loader](https://github.com/mlzxy/rla-wm/blob/6f19048758699bf9a152eaed5ac6dbf1caa07c18/utils/dino.py#L77-L106) calls `AutoConfig.from_pretrained` and `AutoModel.from_pretrained` for the Meta repository. On several loading failures it substitutes DINOv2-small. That fallback must be replaced with a hard failure in any separately versioned reproduction adapter. Randomly instantiating DINOv3 from its architecture/config cannot recover the missing pretrained parameters; loading RLA encoder tensors into it cannot do so either.

### Direct checkpoint-metadata inspection

For each of the eight IWS component `.pt` files, bounded HTTP Range requests fetched the ZIP end record, central directory, `data.pkl` header and `data.pkl` only. All responses were required to be HTTP 206 with the exact requested content range; a full response would have been rejected before payload reading. Pickle opcodes were inspected with `pickletools.genops`, **not executed or unpickled**. Total checkpoint metadata transferred: **301,216 bytes**. No tensor storage entries were requested.

| Component group | Files inspected | Weight/bias names per file | Observed namespace examples |
|---|---:|---:|---|
| IWS RGB decoder | 1 | 102 | `input_proj.*`, `out_conv.*` |
| PushT, Box, Rope flow models | 3 | 278 each | `qpos_encoders.*`, `cond_blocks.*`, `flow_out_proj.*` |
| Shared IWS / PushT RLA encoders | 2 | 78 each | `input_layer.*`, `blocks.*`, `out_layer.*` |
| Shared IWS / PushT RLA decoders | 2 | 80 each | `token_proj.*`, `input_layer.*`, `out_layer.*` |

These are individual module state dictionaries with the namespaces expected by their released configurations. None contains DINOv3/extractor, image patch-embedding, or register-token namespaces. Together with the separate extractor construction and checkpoint writer, this rules out the proposed bundled-backbone shortcut for these artifacts. This is metadata verification, not a full tensor-integrity check or successful model load.

The smallest *unchanged official PushT pipeline* requires these four public component files, totaling **3,557,510,852 bytes**, in addition to the approved DINOv3-L backbone and dependency environment:

| Path under the pinned model release | Bytes | Published LFS SHA256 |
|---|---:|---|
| `rla-wm/iws/pusht/20260409_00-49-21/ckpts/flow_model_step0085000.pt` | 2,084,731,091 | `e94fbdbf8f1bcf33d003ec28af70a072e729941309586f3976502eb790c1c66e` |
| `rla/iws_pusht/20260406_13-33-29/ckpts/encoder_step0067500.pt` | 609,054,925 | `ce6ab17d7822357895e26bd27d2be45d57a27e239513807cce0bfdeccc90fd78` |
| `rla/iws_pusht/20260406_13-33-29/ckpts/decoder_step0067500.pt` | 613,257,761 | `5def176644c5dc02c6ece0761db00335b475aa1f7fed39c4ee982d02e9731a4a` |
| `dino-to-image_unet/iws/20260402_18-21-21/ckpts/decoder_step0050000.pt` | 250,467,075 | `3cad40f46bef9b131ff9bf98d4c36cdd779e92b5d28469e4ea9981eed45c3850` |

Hashes above are publisher metadata, not hashes of locally downloaded full weights. The model release's license label should not be presented as replacing Meta's separate backbone conditions.

## Smallest legitimate comparison track

1. **Prepare one-task frozen-baseline reproduction first: PushT.** Secure approved exact-backbone access without substitutions; use an isolated environment and only the four listed component files. Make a separately documented compatibility adapter for the predictor's nonexistent `DinoLatentActionFlowTrainer` import, mapping to the released `RLAWMTrainer` after strict config/state checks. Disable backbone fallback, pin model revision/preprocessing, and preserve 30 Euler steps, sampling seed, one initial image, task identity and all supplied command rows. Resource/profile checks may use authorized training examples only. The repository's training RAM recommendation is not a measured inference requirement.
2. **Keep the current evaluation lock.** Run the official 200 PushT handles only after the registered internal-development campaign and evaluation contract are frozen and the root opens the reserved split. Preserve the exact upstream endpoint `s+59`, 60 command rows and four disjoint 15-row rollout chunks. The [existing temporal audit](../real_video_development/iws_temporal_semantics_review.md) documents the unresolved chunk-boundary convention; do not silently repair it. No physical-second claim follows from these indices.
3. **Label the first outcome an external reproduction/reference, not a head-to-head win.** RLA-WM reports DINOv3-token L1 and decoded-image LPIPS/SSIM. Our current compact model predicts DINOv2-small 4×4 features and reports feature errors. Different encoder coordinates, pooling, normalization and losses make their native numbers incomparable; our current model does not output RGB for LPIPS/SSIM. The upstream model also uses masked 512-pixel inputs, versus our full-frame 224-pixel encoder preprocessing.
4. **Register a common-representation comparison separately before making a superiority claim.** The closest architectural test would train a ShiftWM variant in the same frozen DINOv3 target coordinates, then use the same feature metric and, if RGB metrics are required, the same fixed image decoder. This is a new model/cache/resource contract, not a checkpoint conversion of the current run. Profile its actual full-shape cost before committing to it. A faster same-base residual/adaptation wrapper around frozen RLA-WM can yield a fair paired RLA-WM-versus-wrapper experiment, but it is a different intervention and must not be renamed as evidence for the already trained compact ShiftWM architecture.

Do not use our internal development split as an unseen external test for released RLA-WM: it is carved from the upstream training trajectories that its released models were configured to train on. The upstream official validation split was also used in its training-time monitoring; any comparison should preserve that benchmark's name rather than claim that it was untouched by every participating method. A faithful released-checkpoint reference can still be useful, with the distinct training and selection budgets disclosed.

**Immediate decision:** continue the registered 27-run compact IWS campaign unchanged. It supplies matched architectural controls, not reproduced external SOTA. In parallel, the next external-baseline dependency is approved DINOv3-L access; the next scientific dependency is a shared evaluation representation. Neither can be solved by relabeling the public RLA encoder or comparing unlike feature errors.

## Reproduction record

Temporary metadata-only audit files are in `/tmp/rla-wm-metadata-audit/`; no credentials were read or saved. The API manifest SHA256 is `9e7b65b98be5bea7d2c98efa658398fd046465e3560cd6d169ff4fa05c8e521c`. The eight-file range-inspection receipt SHA256 is `0026313bfc84b0fde7db720fb7dd02dd08afadd54dc5007d5e8f8a6741de449d`.

| Pinned local source | SHA256 |
|---|---|
| `src/trainers/rla_wm_trainer.py` | `faa57d9834e41b614e2b9b07540ab9b9bbc76b25483529c88b36b7a47625ff19` |
| `src/trainers/basic.py` | `6ec44b3392c3c2721e3ab2bdf3767f653b59eb3927734cc849e67fee71c099e5` |
| `utils/dino.py` | `d59ccb1e55d521f98c0b945a308d7436a62f5ff258972b9aa8af92597dd583b1` |
| `eval/predictors/rla_wm_predictor_iws.py` | `54aa0de48ba8430580d5dd41979d8f8dc0996cb5ce930044aa9cbb1b07ae649a` |

Paths in the source table are relative to `external/rla-wm`. The only repository file created by this audit is this report.
