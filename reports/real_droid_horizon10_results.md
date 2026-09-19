# Matched horizon-ten training control

Status: completed; 12/12 full 30-epoch runs.

Original h5 models selected window-weighted all-five validation; new models preserve window weighting and select all-ten validation. The training and matching selection horizon change together. Equal-episode evaluation is unchanged; the auxiliary equal-episode journal metric never selects checkpoints.

| Comparison | Horizon | Metric | First MSE | Second MSE | First relative reduction | Paired difference 95% interval |
|---|---:|---|---:|---:|---:|---|
| ShiftWM (ours) vs Framewise, both h10-trained | 5 | h5_standardized_mse | 0.158302 | 0.158660 | +0.226% | [-0.000517, -0.000221] |
| ShiftWM (ours) vs Framewise, both h10-trained | 5 | mean_standardized_mse | 0.112711 | 0.112857 | +0.129% | [-0.000219, -0.000083] |
| ShiftWM (ours) vs Framewise, both h10-trained | 10 | h10_standardized_mse | 0.213422 | 0.214289 | +0.405% | [-0.001335, -0.000490] |
| ShiftWM (ours) vs Framewise, both h10-trained | 10 | mean_standardized_mse | 0.152223 | 0.152607 | +0.252% | [-0.000585, -0.000218] |
| framewise: h10-trained vs original h5-trained | 5 | h5_standardized_mse | 0.156300 | 0.156712 | +0.263% | [-0.000807, -0.000020] |
| framewise: h10-trained vs original h5-trained | 5 | mean_standardized_mse | 0.109622 | 0.109717 | +0.087% | [-0.000277, +0.000070] |
| framewise: h10-trained vs original h5-trained | 10 | h10_standardized_mse | 0.214289 | 0.216903 | +1.205% | [-0.003834, -0.001485] |
| framewise: h10-trained vs original h5-trained | 10 | mean_standardized_mse | 0.152607 | 0.153459 | +0.555% | [-0.001350, -0.000393] |
| constant_dynamics: h10-trained vs original h5-trained | 5 | h5_standardized_mse | 0.156272 | 0.156638 | +0.233% | [-0.000756, +0.000023] |
| constant_dynamics: h10-trained vs original h5-trained | 5 | mean_standardized_mse | 0.109610 | 0.109681 | +0.065% | [-0.000254, +0.000094] |
| constant_dynamics: h10-trained vs original h5-trained | 10 | h10_standardized_mse | 0.214233 | 0.216746 | +1.159% | [-0.003752, -0.001376] |
| constant_dynamics: h10-trained vs original h5-trained | 10 | mean_standardized_mse | 0.152579 | 0.153382 | +0.523% | [-0.001304, -0.000341] |
| factorized: h10-trained vs original h5-trained | 5 | h5_standardized_mse | 0.155996 | 0.156234 | +0.152% | [-0.000899, +0.000606] |
| factorized: h10-trained vs original h5-trained | 5 | mean_standardized_mse | 0.109505 | 0.109379 | -0.116% | [-0.000215, +0.000594] |
| factorized: h10-trained vs original h5-trained | 10 | h10_standardized_mse | 0.213422 | 0.219507 | +2.772% | [-0.009359, -0.003088] |
| factorized: h10-trained vs original h5-trained | 10 | mean_standardized_mse | 0.152223 | 0.153851 | +1.058% | [-0.002725, -0.000804] |
| action_free: h10-trained vs original h5-trained | 5 | h5_standardized_mse | 0.157093 | 0.157491 | +0.252% | [-0.000673, -0.000101] |
| action_free: h10-trained vs original h5-trained | 5 | mean_standardized_mse | 0.110016 | 0.110105 | +0.081% | [-0.000223, +0.000052] |
| action_free: h10-trained vs original h5-trained | 10 | h10_standardized_mse | 0.214963 | 0.217435 | +1.137% | [-0.003233, -0.001684] |
| action_free: h10-trained vs original h5-trained | 10 | mean_standardized_mse | 0.153224 | 0.154028 | +0.522% | [-0.001141, -0.000458] |

Report every mode/seed. H10 benefit requires negative upper paired 95% CI for new-minus-original all-ten mean error; compare ours with equally h10-trained Framewise separately. Validation evidence only; no new test claim.

Intervals resample training seeds and recording-session clusters (10,000 draws). Exploratory, unadjusted for multiple comparisons. Signed changes and negative results are retained.
