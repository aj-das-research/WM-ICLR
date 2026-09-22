# Post-hoc planning robustness

post-hoc diagnostic on revealed current-spatial simulator tests; no model/endpoint reselection

- pusht transport vs autoregressive: -0.33 pp; seed differences [1.0, -4.0, 2.0]; two-sided p=1, Holm(8)=1.
- pusht transport vs bounded_additive: +0.67 pp; seed differences [-2.0, -4.0, 8.0]; two-sided p=0.886701, Holm(8)=1.
- pusht unbounded_transport vs transport: -1.00 pp; seed differences [-3.0, 5.0, -5.0]; two-sided p=0.690633, Holm(8)=1.
- pusht unbounded_transport vs autoregressive: -1.33 pp; seed differences [-2.0, 1.0, -3.0]; two-sided p=0.679633, Holm(8)=1.
- reacher transport vs autoregressive: -53.00 pp; seed differences [-47.0, -54.0, -57.99999999999999]; two-sided p=9.9999e-06, Holm(8)=7.99992e-05.
- reacher transport vs bounded_additive: -5.67 pp; seed differences [5.0, -5.0, -17.0]; two-sided p=0.0938791, Holm(8)=0.563274.
- reacher unbounded_transport vs transport: -1.67 pp; seed differences [-10.0, -1.0, 6.0]; two-sided p=0.641174, Holm(8)=1.
- reacher unbounded_transport vs autoregressive: -54.67 pp; seed differences [-56.99999999999999, -55.00000000000001, -52.0]; two-sided p=9.9999e-06, Holm(8)=7.99992e-05.

conditional on the three fitted seeds; symmetric cluster effects required for sign-flip inference; sensitivity analysis, not confirmatory evidence

Terminal failure decomposition (counts are correlated seed-case records, not independent episodes):
{
  "pusht": {
    "autoregressive": {
      "failed_seed_case_records": 265,
      "all_seed_case_records": 300,
      "categories": {
        "position_only": 142,
        "angle_only": 2,
        "both": 121
      },
      "seed_success_percent": [
        9.0,
        11.0,
        15.0
      ]
    },
    "bounded_additive": {
      "failed_seed_case_records": 268,
      "all_seed_case_records": 300,
      "categories": {
        "position_only": 124,
        "angle_only": 1,
        "both": 143
      },
      "seed_success_percent": [
        12.0,
        11.0,
        9.0
      ]
    },
    "transport": {
      "failed_seed_case_records": 266,
      "all_seed_case_records": 300,
      "categories": {
        "position_only": 130,
        "angle_only": 0,
        "both": 136
      },
      "seed_success_percent": [
        10.0,
        7.000000000000001,
        17.0
      ]
    },
    "unbounded_transport": {
      "failed_seed_case_records": 269,
      "all_seed_case_records": 300,
      "categories": {
        "position_only": 129,
        "angle_only": 0,
        "both": 140
      },
      "seed_success_percent": [
        7.000000000000001,
        12.0,
        12.0
      ]
    }
  },
  "reacher": {
    "autoregressive": {
      "failed_seed_case_records": 60,
      "all_seed_case_records": 300,
      "categories": {
        "joint1_only": 6,
        "joint2_only": 22,
        "both_joints": 32
      },
      "seed_success_percent": [
        79.0,
        81.0,
        80.0
      ]
    },
    "bounded_additive": {
      "failed_seed_case_records": 202,
      "all_seed_case_records": 300,
      "categories": {
        "joint1_only": 5,
        "joint2_only": 29,
        "both_joints": 168
      },
      "seed_success_percent": [
        27.0,
        32.0,
        39.0
      ]
    },
    "transport": {
      "failed_seed_case_records": 219,
      "all_seed_case_records": 300,
      "categories": {
        "joint1_only": 7,
        "joint2_only": 36,
        "both_joints": 176
      },
      "seed_success_percent": [
        32.0,
        27.0,
        22.0
      ]
    },
    "unbounded_transport": {
      "failed_seed_case_records": 224,
      "all_seed_case_records": 300,
      "categories": {
        "joint1_only": 13,
        "joint2_only": 38,
        "both_joints": 173
      },
      "seed_success_percent": [
        22.0,
        26.0,
        28.000000000000004
      ]
    }
  }
}
