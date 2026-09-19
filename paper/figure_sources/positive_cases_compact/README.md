# Two additional positive Reacher cases

Reproduce with `python paper/scripts/render_positive_cases_compact.py`.
Runtime inputs are the renderer, data.json, manifest.json, caption.tex and 24 public PNGs. Historical experiment and NPZ paths in data.json are provenance only. No raw experiment, checkpoint, simulator, model or network access is required.

The historical context model at training seed 0 is distinct from the current spatial decoder. The displayed original cases C/D, episodes 2031008 and 2031011, now have labels (a)/(b). No case was reselected. The other positive-discordant cases, PushT 2031024 and Reacher 2031004, remain in technical and outcome figures. The complete inventory and all three methods' stored observations and criteria are retained in the pack. Shared context fails on all four. Eligible ShiftWM-only versus Framewise-only successes are 1:1 on PushT and 3:8 on Reacher; positive cases do not estimate frequency.

Images show the common goal, matched native call 20 and actual endpoints: ours 21/23, Framewise 50/50. Methods share ten support actions and a 50-action allowance. Details use exactly the original per-case crop; full-goal overviews locate it. Purple dashed goal contours are display-only annotations, never scoring or prediction inputs. Public PNGs preserve full 224×224 observations. PDF image samples are unchanged source samples; vector viewports apply the declared common crop.

Dots show already saved, validated exact-replay raw joint errors. Success requires both unwrapped errors strictly below 0.05 radians. Horizontal strokes pair methods on the same joint at their respective endpoints; they are not trajectories, uncertainty or confidence intervals. L2 is diagnostic only: 0.060 can pass if both joints remain below 0.05. Both axes share 0–0.4 radians. Circle/square markers preserve method identity in grayscale.

The source produces editable quantitative, text and contour layers with raster observations. The canvas is 5.5×2.917 inches, with labels of at least 8 points. Rendering changes no manuscript or old assets. The original tall positive gallery stays archived.
