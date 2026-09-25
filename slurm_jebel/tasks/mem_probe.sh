# Peak GPU memory of one training step (fwd+bwd, bf16 autocast) per arm and batch size, excluding resident features.
python - <<'PY'
import json, torch
from shiftwm.v2.train import build, loss_fn
cfg = json.load(open("configs/v2/droid_base.json"))
for arm in ["shiftwm", "direct", "ar", "ar_tf"]:
    for bs in [48, 64, 128]:
        c = json.loads(json.dumps(cfg)); c["model"]["arm"] = arm
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        try:
            m = build(c, 16, 384, 35).cuda()
            h = torch.randn(bs, 3, 256, 384, device="cuda"); p = torch.randn(bs, 2, 35, device="cuda")
            f = torch.randn(bs, 10, 35, device="cuda"); t = torch.randn(bs, 10, 256, 384, device="cuda")
            with torch.autocast("cuda", dtype=torch.bfloat16):
                loss, _ = loss_fn(m, (h, p, f, t), c)
            loss.backward()
            print(json.dumps({"arm": arm, "bs": bs, "peak_GB": round(torch.cuda.max_memory_allocated() / 1e9, 2)}), flush=True)
        except Exception as e:
            print(json.dumps({"arm": arm, "bs": bs, "error": str(e)[:200]}), flush=True)
        del m
PY
