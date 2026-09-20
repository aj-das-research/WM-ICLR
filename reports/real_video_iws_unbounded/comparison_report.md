# IWS unbounded-innovation ablation — complete exploratory development report

All nine new runs and all 27 frozen common-CPU v1 comparison runs passed. No reserved data were used. Lower errors are better; positive relative gains favor the unbounded ablation. This is a component study after development, not a new algorithm or a confirmatory result.

Means average windows within trajectory, trajectories and three seeds equally. H labels count command rows; the target is stored offset H−1. Only H60 has paired 10,000-draw seed×trajectory intervals (seed173), unadjusted exploratory95% intervals. Prefix summaries are descriptive.

| Task | Metric | H | Unbounded | Bounded | Additive | AR | Persistence |
|---|---|---:|---:|---:|---:|---:|---:|
| pusht | standardized_mse | 15 | 0.201687725 | 0.210807509 | 0.218379096 | 0.217426976 | 0.395410515 |
| pusht | standardized_mse | 30 | 0.224299032 | 0.240489392 | 0.246789765 | 0.242606663 | 0.514163964 |
| pusht | standardized_mse | 45 | 0.233372119 | 0.251740049 | 0.25777886 | 0.252012121 | 0.54760719 |
| pusht | standardized_mse | 60 | 0.241617656 | 0.261116483 | 0.267429044 | 0.25999904 | 0.571659275 |
| pusht | standardized_mae | 15 | 0.344345316 | 0.351299266 | 0.358353831 | 0.358415255 | 0.4697313 |
| pusht | standardized_mae | 30 | 0.363543625 | 0.375172217 | 0.381637194 | 0.379434876 | 0.536083788 |
| pusht | standardized_mae | 45 | 0.370615281 | 0.383572164 | 0.389930366 | 0.386755428 | 0.55388988 |
| pusht | standardized_mae | 60 | 0.376721381 | 0.390368521 | 0.396836722 | 0.392733874 | 0.566374872 |
| pusht | raw_dinov2_l1 | 15 | 0.361670333 | 0.369260679 | 0.376827461 | 0.377613375 | 0.497089243 |
| pusht | raw_dinov2_l1 | 30 | 0.382386316 | 0.394923522 | 0.402079051 | 0.400697212 | 0.56964975 |
| pusht | raw_dinov2_l1 | 45 | 0.390144372 | 0.404092439 | 0.411175428 | 0.40881921 | 0.589182441 |
| pusht | raw_dinov2_l1 | 60 | 0.396829597 | 0.411516693 | 0.418766904 | 0.415425689 | 0.602920481 |
| pusht | feature_cosine_distance | 15 | 0.0343059196 | 0.0360351337 | 0.0373691627 | 0.0373461743 | 0.0699522883 |
| pusht | feature_cosine_distance | 30 | 0.0384756548 | 0.0415146922 | 0.0426105249 | 0.042086421 | 0.0929204439 |
| pusht | feature_cosine_distance | 45 | 0.0402189188 | 0.043673481 | 0.0447019199 | 0.0439060956 | 0.0993364713 |
| pusht | feature_cosine_distance | 60 | 0.0418254042 | 0.0455005829 | 0.0465692102 | 0.0454513259 | 0.103963503 |
| bimanual_box | standardized_mse | 15 | 0.224880452 | 0.233574848 | 0.2435728 | 0.240846293 | 0.398993457 |
| bimanual_box | standardized_mse | 30 | 0.248919152 | 0.268259793 | 0.275996428 | 0.267224006 | 0.540745752 |
| bimanual_box | standardized_mse | 45 | 0.257132231 | 0.282030487 | 0.287983257 | 0.275244654 | 0.600071427 |
| bimanual_box | standardized_mse | 60 | 0.263284772 | 0.290926916 | 0.296560395 | 0.279410478 | 0.630393247 |
| bimanual_box | standardized_mae | 15 | 0.360775796 | 0.366676992 | 0.37514413 | 0.374096093 | 0.468619467 |
| bimanual_box | standardized_mae | 30 | 0.381016702 | 0.393597226 | 0.40135836 | 0.39568032 | 0.54868039 |
| bimanual_box | standardized_mae | 45 | 0.387715653 | 0.403886078 | 0.410623735 | 0.402261945 | 0.580321293 |
| bimanual_box | standardized_mae | 60 | 0.392340048 | 0.410229734 | 0.41676824 | 0.405635544 | 0.596194429 |
| bimanual_box | raw_dinov2_l1 | 15 | 0.375149572 | 0.381438984 | 0.390416058 | 0.389783891 | 0.48709386 |
| bimanual_box | raw_dinov2_l1 | 30 | 0.396668954 | 0.409882542 | 0.418524727 | 0.41325308 | 0.571894738 |
| bimanual_box | raw_dinov2_l1 | 45 | 0.403869764 | 0.420822184 | 0.428545694 | 0.420558811 | 0.605574166 |
| bimanual_box | raw_dinov2_l1 | 60 | 0.408862939 | 0.427627165 | 0.435190524 | 0.42438561 | 0.622460518 |
| bimanual_box | feature_cosine_distance | 15 | 0.0372187326 | 0.0387707926 | 0.0404989151 | 0.0401105015 | 0.0654237451 |
| bimanual_box | feature_cosine_distance | 30 | 0.0414438841 | 0.0448639184 | 0.0462412863 | 0.0448394351 | 0.0897842276 |
| bimanual_box | feature_cosine_distance | 45 | 0.0429395955 | 0.0473450754 | 0.0484263809 | 0.0463503839 | 0.100109604 |
| bimanual_box | feature_cosine_distance | 60 | 0.0440622356 | 0.0489716822 | 0.0499759389 | 0.0471473889 | 0.10533879 |
| bimanual_rope | standardized_mse | 15 | 0.171757479 | 0.180427459 | 0.186671035 | 0.182549888 | 0.357064089 |
| bimanual_rope | standardized_mse | 30 | 0.187733719 | 0.206140947 | 0.209353115 | 0.201693112 | 0.514668264 |
| bimanual_rope | standardized_mse | 45 | 0.193396302 | 0.217100072 | 0.217845796 | 0.208300688 | 0.587493905 |
| bimanual_rope | standardized_mse | 60 | 0.197045831 | 0.222913001 | 0.222888095 | 0.211975513 | 0.61713679 |
| bimanual_rope | standardized_mae | 15 | 0.31710367 | 0.324212885 | 0.330886748 | 0.327974732 | 0.445041164 |
| bimanual_rope | standardized_mae | 30 | 0.332102428 | 0.3463681 | 0.351592278 | 0.34565019 | 0.53466512 |
| bimanual_rope | standardized_mae | 45 | 0.337224805 | 0.355258123 | 0.358971284 | 0.351537252 | 0.571381926 |
| bimanual_rope | standardized_mae | 60 | 0.340391591 | 0.359898889 | 0.363139294 | 0.354677859 | 0.585841472 |
| bimanual_rope | raw_dinov2_l1 | 15 | 0.321610037 | 0.329047202 | 0.336340787 | 0.333946032 | 0.454649757 |
| bimanual_rope | raw_dinov2_l1 | 30 | 0.337076829 | 0.351737813 | 0.358056958 | 0.352753437 | 0.54938773 |
| bimanual_rope | raw_dinov2_l1 | 45 | 0.342363562 | 0.360850887 | 0.365820604 | 0.359076936 | 0.588454907 |
| bimanual_rope | raw_dinov2_l1 | 60 | 0.345662031 | 0.365637773 | 0.370140988 | 0.362410206 | 0.603723519 |
| bimanual_rope | feature_cosine_distance | 15 | 0.0263451831 | 0.0277832325 | 0.0289136466 | 0.0283699551 | 0.0569075287 |
| bimanual_rope | feature_cosine_distance | 30 | 0.0289366184 | 0.0319293584 | 0.0326907373 | 0.0316328331 | 0.0846410579 |
| bimanual_rope | feature_cosine_distance | 45 | 0.0298915355 | 0.033732757 | 0.0341538287 | 0.0328176024 | 0.0977307035 |
| bimanual_rope | feature_cosine_distance | 60 | 0.0305226005 | 0.0347083585 | 0.0350147456 | 0.0334713119 | 0.102942643 |

| Task | Metric at H60 | Comparator | Unbounded−comparator | Relative gain (%) | Paired95% gain interval (%) |
|---|---|---|---:|---:|---|
| pusht | standardized_mse | bounded_spatial_mix | -0.0194988275 | +7.467482 | [+7.083474, +7.852452] |
| pusht | standardized_mse | anchored_additive | -0.0258113883 | +9.651677 | [+9.348538, +9.952393] |
| pusht | standardized_mse | autoregressive | -0.018381384 | +7.069789 | [+6.155804, +7.944196] |
| pusht | standardized_mse | persistence | -0.33004162 | +57.733974 | [+56.659408, +58.753418] |
| pusht | standardized_mae | bounded_spatial_mix | -0.0136471405 | +3.495963 | [+3.305684, +3.685522] |
| pusht | standardized_mae | anchored_additive | -0.0201153417 | +5.068921 | [+4.904487, +5.230699] |
| pusht | standardized_mae | autoregressive | -0.0160124932 | +4.077187 | [+3.663876, +4.485733] |
| pusht | standardized_mae | persistence | -0.189653492 | +33.485506 | [+32.633563, +34.306738] |
| pusht | raw_dinov2_l1 | bounded_spatial_mix | -0.014687096 | +3.569016 | [+3.373984, +3.763248] |
| pusht | raw_dinov2_l1 | anchored_additive | -0.0219373068 | +5.238548 | [+5.064938, +5.408894] |
| pusht | raw_dinov2_l1 | autoregressive | -0.0185960922 | +4.476394 | [+4.045712, +4.900649] |
| pusht | raw_dinov2_l1 | persistence | -0.206090884 | +34.182100 | [+33.295950, +35.037334] |
| pusht | feature_cosine_distance | bounded_spatial_mix | -0.0036751787 | +8.077212 | [+7.672126, +8.486952] |
| pusht | feature_cosine_distance | anchored_additive | -0.00474380599 | +10.186572 | [+9.852647, +10.515166] |
| pusht | feature_cosine_distance | autoregressive | -0.00362592164 | +7.977593 | [+6.917570, +9.002938] |
| pusht | feature_cosine_distance | persistence | -0.0621380986 | +59.769147 | [+58.595018, +60.868662] |
| bimanual_box | standardized_mse | bounded_spatial_mix | -0.0276421434 | +9.501405 | [+8.917508, +10.077958] |
| bimanual_box | standardized_mse | anchored_additive | -0.0332756227 | +11.220521 | [+10.852758, +11.572206] |
| bimanual_box | standardized_mse | autoregressive | -0.0161257058 | +5.771332 | [+5.074891, +6.475817] |
| bimanual_box | standardized_mse | persistence | -0.367108475 | +58.234836 | [+57.099152, +59.301868] |
| bimanual_box | standardized_mae | bounded_spatial_mix | -0.0178896858 | +4.360894 | [+4.097242, +4.621946] |
| bimanual_box | standardized_mae | anchored_additive | -0.0244281923 | +5.861337 | [+5.667951, +6.046710] |
| bimanual_box | standardized_mae | autoregressive | -0.0132954961 | +3.277695 | [+2.983225, +3.586532] |
| bimanual_box | standardized_mae | persistence | -0.203854382 | +34.192601 | [+33.315537, +35.045513] |
| bimanual_box | raw_dinov2_l1 | bounded_spatial_mix | -0.0187642264 | +4.387987 | [+4.124898, +4.646582] |
| bimanual_box | raw_dinov2_l1 | anchored_additive | -0.0263275855 | +6.049669 | [+5.855394, +6.235568] |
| bimanual_box | raw_dinov2_l1 | autoregressive | -0.0155226714 | +3.657681 | [+3.367573, +3.963208] |
| bimanual_box | raw_dinov2_l1 | persistence | -0.213597579 | +34.315041 | [+33.424279, +35.173496] |
| bimanual_box | feature_cosine_distance | bounded_spatial_mix | -0.00490944656 | +10.025072 | [+9.431298, +10.610496] |
| bimanual_box | feature_cosine_distance | anchored_additive | -0.00591370327 | +11.833101 | [+11.470839, +12.180521] |
| bimanual_box | feature_cosine_distance | autoregressive | -0.00308515323 | +6.543635 | [+5.832882, +7.271880] |
| bimanual_box | feature_cosine_distance | persistence | -0.0612765548 | +58.170931 | [+56.976963, +59.285239] |
| bimanual_rope | standardized_mse | bounded_spatial_mix | -0.0258671702 | +11.604155 | [+11.114614, +12.096994] |
| bimanual_rope | standardized_mse | anchored_additive | -0.0258422641 | +11.594277 | [+11.183965, +11.946300] |
| bimanual_rope | standardized_mse | autoregressive | -0.0149296822 | +7.043116 | [+6.670164, +7.412283] |
| bimanual_rope | standardized_mse | persistence | -0.420090959 | +68.070964 | [+67.257699, +68.845365] |
| bimanual_rope | standardized_mae | bounded_spatial_mix | -0.0195072984 | +5.420216 | [+5.192177, +5.650838] |
| bimanual_rope | standardized_mae | anchored_additive | -0.0227477029 | +6.264181 | [+6.054312, +6.447642] |
| bimanual_rope | standardized_mae | autoregressive | -0.0142862684 | +4.027956 | [+3.843259, +4.211158] |
| bimanual_rope | standardized_mae | persistence | -0.245449882 | +41.896980 | [+41.163457, +42.596135] |
| bimanual_rope | raw_dinov2_l1 | bounded_spatial_mix | -0.0199757411 | +5.463260 | [+5.237791, +5.688741] |
| bimanual_rope | raw_dinov2_l1 | anchored_additive | -0.0244789569 | +6.613414 | [+6.403159, +6.797396] |
| bimanual_rope | raw_dinov2_l1 | autoregressive | -0.0167481744 | +4.621331 | [+4.438784, +4.806625] |
| bimanual_rope | raw_dinov2_l1 | persistence | -0.258061488 | +42.744978 | [+41.974937, +43.472823] |
| bimanual_rope | feature_cosine_distance | bounded_spatial_mix | -0.00418575799 | +12.059798 | [+11.575761, +12.543537] |
| bimanual_rope | feature_cosine_distance | anchored_additive | -0.0044921451 | +12.829295 | [+12.412352, +13.191912] |
| bimanual_rope | feature_cosine_distance | autoregressive | -0.0029487114 | +8.809668 | [+8.381994, +9.233242] |
| bimanual_rope | feature_cosine_distance | persistence | -0.0724200426 | +70.349896 | [+69.451868, +71.182334] |

| Equal-task macro, H60 metric | Comparator | Mean relative gain (%) | Paired95% interval (%) |
|---|---|---:|---|
| standardized_mse | bounded_spatial_mix | +9.524347 | [+9.237260, +9.803204] |
| standardized_mse | anchored_additive | +10.822159 | [+10.625109, +11.025442] |
| standardized_mse | autoregressive | +6.628079 | [+6.214036, +7.069254] |
| standardized_mse | persistence | +61.346591 | [+60.755601, +61.896960] |
| standardized_mae | bounded_spatial_mix | +4.425691 | [+4.292537, +4.556726] |
| standardized_mae | anchored_additive | +5.731480 | [+5.628284, +5.836618] |
| standardized_mae | autoregressive | +3.794279 | [+3.601599, +4.001740] |
| standardized_mae | persistence | +36.525029 | [+36.053036, +36.982657] |
| raw_dinov2_l1 | bounded_spatial_mix | +4.473421 | [+4.341334, +4.603970] |
| raw_dinov2_l1 | anchored_additive | +5.967210 | [+5.861906, +6.073714] |
| raw_dinov2_l1 | autoregressive | +4.251802 | [+4.054788, +4.465106] |
| raw_dinov2_l1 | persistence | +37.080706 | [+36.593482, +37.554311] |
| feature_cosine_distance | bounded_spatial_mix | +10.054028 | [+9.766088, +10.338765] |
| feature_cosine_distance | anchored_additive | +11.616323 | [+11.410257, +11.825477] |
| feature_cosine_distance | autoregressive | +7.776965 | [+7.308956, +8.288100] |
| feature_cosine_distance | persistence | +62.763325 | [+62.136437, +63.349168] |
