# Spatial component follow-up: independent prelaunch review

**Passed.** Reviewed all new model, trainer, evaluator, campaign, statistical, inference-export, scheduler, test and protocol sources. The JSON receipt binds their exact hashes.

The two new arms complete the missing mixing × bounding cells with six new full30-epoch runs and six already revealed controls. The recipe preserves the original data, normalization, information budget, horizon, optimizer and selection criterion. All20 edge/interaction contrasts are retained. This is explicitly exploratory validation development, not held-out confirmation or SOTA evidence.

One gate gap was fixed before signoff: a self-consistent executed package was not explicitly compared with its registered scientific recipe. Both control registration and final collection now check this equality; the targeted LR/batch/epoch/mode tamper test passed independently.

Independent synthetic-input probes used genuine trained control tensors only as an engineering check. Both new decoder formulas agree with independently reconstructed formulas within2.39e-7. Earlier forecasts are exactly invariant to changed later commands; gradients from query images and later commands are zero, while permitted command gradients are nonzero. The bounded arm obeys its bound; the unbounded arm exceeds it in a deliberate stress case. Horizon truncation has at most4.77e-7 FP32 numerical differences and should not be described as bitwise invariant.

A separate count-weighted bootstrap implementation reproduced all20 effects and intervals on unequal-size synthetic sessions, with maximum discrepancy6.7e-16. This is statistical implementation verification, not a benchmark result.

The implementing agent reports83 passing tests (34 new and49 existing), compilation and scheduler syntax checks, and successful strict audit of all six actual controls. I independently reran the new executed-recipe regression test and the mechanism/statistical probes.

The reviewed scheduler uses two ws-ia serial lanes, one allocated GPU and eight CPUs per lane, unchanged batch128/full30 training, and a CPU-only finalizer. Allocated free memory is recorded and checked before training. No GPU visibility override or outcome-dependent fallback is permitted.

No blocking issue remains. Continue to bind the launched registry/job identities in a separate operational receipt. Keep mixing-package capacity/initialization caveats and all null/negative outcomes visible. Export-directory promotion followed by a process interruption may still need explicit recovery rather than an automatic overwrite.
