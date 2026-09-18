# Causal support-reliability diagnostic results

Original validation only; exploratory development evidence. All arms share the same 13-frame prefix and eligible windows. Frozen equally calibrated donors; no new neural training.

| Arm | h5 MSE | h10 MSE | h5 change vs global-prior blend | h5 paired MSE difference 95% interval |
|---|---:|---:|---:|---|
| framewise | 0.15247186 | 0.21374215 | -0.464% | [+0.00025118, +0.00139377] |
| factorized | 0.15176811 | 0.21362396 | +0.000% | [+0.00000000, +0.00000000] |
| equal_blend | 0.15151322 | 0.21178491 | +0.168% | [-0.00068844, +0.00014062] |
| train_prior_blend | 0.15176811 | 0.21362396 | +0.000% | [+0.00000000, +0.00000000] |
| train_query_blend | 0.15176811 | 0.21362396 | +0.000% | [+0.00000000, +0.00000000] |
| unshrunk_local | 0.15160572 | 0.21226556 | +0.107% | [-0.00054360, +0.00011958] |
| shrunk_local | 0.15175676 | 0.21357716 | +0.007% | [-0.00002378, +0.00000256] |
| one_step_local | 0.15176066 | 0.21359542 | +0.005% | [-0.00001751, +0.00000375] |
| shuffled_local | 0.15176150 | 0.21358964 | +0.004% | [-0.00001439, +0.00000217] |
| persistence | 0.15570474 | 0.21590753 | -2.594% | [+0.00229694, +0.00573791] |
| constant_velocity | 1.91607495 | 6.87981302 | -1162.502% | [+1.56707672, +1.96637054] |

Prespecified promotion: **FAIL — do not promote**.

| Criterion | Passed |
|---|---|
| at_least_one_percent_versus_stronger_donor | False |
| at_least_one_percent_versus_train_prior_blend | False |
| at_least_one_percent_versus_train_query_blend | False |
| h5_ci_vs_prior_strictly_negative | False |
| all_seeds_improve_h5_vs_prior | False |
| h10_no_worse_than_prior | True |

All intervals resample recording sessions and training seeds and are unadjusted development intervals. Prefix-fit improvements are not guarantees on future queries. Shuffled gates are an observational mechanism diagnostic, not a deployment method.

## Training-only fitted parameters

| Seed | Prefix prior | Prefix kappa | One-step prior | One-step kappa | Query-fitted constant |
|---|---:|---:|---:|---:|---:|
| 0 | 1.000000 | 100 | 1.000000 | 100 | 1.000000 |
| 1 | 1.000000 | 100 | 1.000000 | 100 | 1.000000 |
| 2 | 1.000000 | 100 | 1.000000 | 100 | 1.000000 |

Complete per-seed/per-episode errors, h10 intervals, length exclusions, gate distributions and artifact identities are retained in the companion JSON and sufficient-statistic caches.

{
  "slurm_job_id": "200201",
  "device": "cpu",
  "cpu_threads": 8,
  "runtime_seconds": 95.56527129909955,
  "maximum_rss_kib": 1015920,
  "validation_features_first_decoded_utc": "2026-09-18T23:39:24.749051+00:00",
  "training_only_lock_sha256": "e46e585db47f8d4deb6a32d2e4fb82b9422fdcc4369396a7c1fa8a736d68cf95",
  "train_population": {
    "split": "train",
    "total_episodes": 851,
    "eligible_episodes": 758,
    "sessions": 295,
    "windows": 6078,
    "excluded_short": [
      {
        "episode_id": "droid-003cb857158280aad4ab174a",
        "session_id": "RAIL/2023-11-08",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-0132439f9f87f8805bfa3915",
        "session_id": "RAIL/2023-07-14",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-0418d46d0eedabc01feab960",
        "session_id": "ILIAD/2023-06-01",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-05fe750a2049d7d03fb35779",
        "session_id": "BVL/2024-02-01",
        "stored_frames": 1,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-06bb152f2a8cd639c307f636",
        "session_id": "ILIAD/2023-07-25",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-0937ec17a25f546e1ed56c0a",
        "session_id": "REAL/2023-05-27",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-0ba95dc1b9c1124875267321",
        "session_id": "AUTOLab/2023-11-30",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-1086692c11bdf433068504b3",
        "session_id": "RAIL/2023-11-03",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-1aee33cecb34b07ffa85ebc1",
        "session_id": "AUTOLab/2023-11-07",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-2068da78130e62ba7a201b51",
        "session_id": "PennPAL/2023-10-30",
        "stored_frames": 9,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-22c7f60f0b6c5a72c9fa37ad",
        "session_id": "AUTOLab/2023-07-21",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-25814a662d5aa2d445b5e08f",
        "session_id": "AUTOLab/2023-11-06",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-26222258ebac0e22d9ff5079",
        "session_id": "CLVR/2023-06-25",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-27529f10270f18011def539a",
        "session_id": "BVL/2024-02-04",
        "stored_frames": 4,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-2c113552243c8aee3f524936",
        "session_id": "AUTOLab/2023-10-13",
        "stored_frames": 16,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-2d429ab2129fee11367f79f2",
        "session_id": "AUTOLab/2023-07-21",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-2ebdd34f28a42fc8f2e9ac9c",
        "session_id": "AUTOLab/2023-11-18",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-2ef7374c55163d840c549a1c",
        "session_id": "AUTOLab/2023-07-14",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-2febf2ab392a0f8a2c30deb9",
        "session_id": "CLVR/2023-06-20",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-372a593de1071a54a8a945e1",
        "session_id": "IPRL/2023-12-19",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-38123070044668dcfab58a7b",
        "session_id": "WEIRD/2023-11-22",
        "stored_frames": 12,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-400eb7faf64cd3805f290112",
        "session_id": "CLVR/2023-06-03",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-42cf7bac39f56d3321d4d20d",
        "session_id": "BVL/2024-02-18",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-48be752c9670650a6de226ab",
        "session_id": "AUTOLab/2023-12-02",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-49bacd40c2aa642bb355466a",
        "session_id": "BVL/2024-02-02",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-49d62ac99a9dc2f4a9c8a279",
        "session_id": "IRIS/2023-03-08",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-4b8e54a49c4ab25eab87eada",
        "session_id": "RAIL/2023-09-29",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5200189ea0befbc7c14edeec",
        "session_id": "RAIL/2023-10-16",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5639be11683b7ba29f8f8198",
        "session_id": "GuptaLab/2023-06-18",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-564d32d20bf8c0fd21dd71c6",
        "session_id": "REAL/2023-08-15",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-59c688a43b080697b604d37f",
        "session_id": "IPRL/2023-12-18",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5c6683efe66ffcb1139f6ee9",
        "session_id": "GuptaLab/2023-06-18",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5d2ca3e6b040f4764cab195c",
        "session_id": "AUTOLab/2023-12-02",
        "stored_frames": 9,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5d8fbacc87fd40108d345aca",
        "session_id": "REAL/2023-07-11",
        "stored_frames": 14,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-60e8fb266e9ca188337ecda9",
        "session_id": "AUTOLab/2023-08-27",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-61e2058c3e921c5b875a4b72",
        "session_id": "IRIS/2023-03-06",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-63625b9a097baa1670a0d3ea",
        "session_id": "ILIAD/2023-12-24",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-69937cabb08b7e9d8106abb7",
        "session_id": "TRI/2024-02-14",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-6c8fb69eba9d8aff7589ab9f",
        "session_id": "AUTOLab/2023-11-17",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-6ca0a6764faf739d75679e48",
        "session_id": "IRIS/2023-03-08",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-6cd04783bb6b5d46a7183cf9",
        "session_id": "AUTOLab/2023-11-29",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7158b67aacaf647e4f016609",
        "session_id": "RAIL/2023-11-30",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-74b036b4e0efabe6ffad8774",
        "session_id": "RAIL/2023-11-03",
        "stored_frames": 14,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-76f28e9f87458c61e0bcbc6c",
        "session_id": "AUTOLab/2023-12-02",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7ac80475682282c7c7b90e29",
        "session_id": "AUTOLab/2023-12-13",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7b1e6558e7089529922c23b7",
        "session_id": "RAIL/2023-11-03",
        "stored_frames": 2,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7c1c02db2065bb58cd26af38",
        "session_id": "TRI/2023-11-09",
        "stored_frames": 13,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7d17c804a8644d92b07a6468",
        "session_id": "BVL/2024-01-30",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7db59f487ee3ad229b15eeae",
        "session_id": "RAIL/2023-11-04",
        "stored_frames": 2,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7fa9dc79a1fe724b6ea5f8c0",
        "session_id": "WEIRD/2023-11-29",
        "stored_frames": 9,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-7fe4284c0ebb5c712fb6eb96",
        "session_id": "RAIL/2023-11-17",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-91b3f4420c2febbd2c212f0f",
        "session_id": "IPRL/2023-12-19",
        "stored_frames": 16,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-934c70ca6185921e8a895afc",
        "session_id": "AUTOLab/2023-10-27",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-96d1502145abeecd5be358a6",
        "session_id": "RAIL/2023-06-30",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-9768f5c1035c695e845a4a2a",
        "session_id": "RAIL/2023-12-02",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-97a20ed9f5cbc38d2851bbff",
        "session_id": "RAIL/2023-07-14",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-9f2dc405b49995e12a75067b",
        "session_id": "ILIAD/2023-07-19",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-a081e3906717c25d3f206603",
        "session_id": "BVL/2024-02-18",
        "stored_frames": 3,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-a51e0619857e1a1ad4c85b70",
        "session_id": "AUTOLab/2023-08-12",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-b1235726565c687119ab3cb9",
        "session_id": "GuptaLab/2023-06-18",
        "stored_frames": 14,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-b7588f71ba072720a1b01a96",
        "session_id": "BVL/2024-02-02",
        "stored_frames": 6,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-b910ff5da83a590a3ca451b6",
        "session_id": "TRI/2024-02-13",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-ba971fd18f84fa9ee3017283",
        "session_id": "TRI/2023-08-07",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-be22c37d35649e6c9fb4e499",
        "session_id": "AUTOLab/2023-11-18",
        "stored_frames": 17,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-c1adda1e7bd6877ceee16b5b",
        "session_id": "WEIRD/2024-01-09",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-c4714dd9e207af540df31d1a",
        "session_id": "CLVR/2023-06-03",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-c6abc8bbeb25ffd290516c05",
        "session_id": "ILIAD/2023-07-27",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-c784ba2d9df2477c4f392a5c",
        "session_id": "TRI/2023-08-07",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-c9f9dfcf3e0b07f5208ae13e",
        "session_id": "GuptaLab/2023-06-18",
        "stored_frames": 17,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-ca53fb277d5ca8a9ff62aa70",
        "session_id": "IPRL/2023-04-21",
        "stored_frames": 2,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-ccaaf1bbaee906e35b2508f5",
        "session_id": "PennPAL/2023-10-18",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-ced11faf282a8d60da01842a",
        "session_id": "AUTOLab/2023-10-27",
        "stored_frames": 13,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-d06564578ca80b53cdad92a3",
        "session_id": "AUTOLab/2023-11-18",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-d2309c86d9bbcf753942a6a1",
        "session_id": "RAIL/2023-06-05",
        "stored_frames": 14,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-d27e7ec65044e1f4feeac090",
        "session_id": "IPRL/2023-06-18",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-d4a83f42f2556347185e0443",
        "session_id": "RAIL/2023-10-14",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-dd87724b4a6192761379d231",
        "session_id": "RAIL/2023-10-09",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-ddac1aac2fe9d419dd40f909",
        "session_id": "TRI/2023-08-07",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-de15657d2e47928da1785763",
        "session_id": "RAIL/2023-11-08",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-df1231912b262d588d7f183a",
        "session_id": "AUTOLab/2023-10-27",
        "stored_frames": 11,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-e124fa7fba9f02a8e74c6090",
        "session_id": "CLVR/2023-06-05",
        "stored_frames": 8,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-e13517ae8f2f8add56fa78ad",
        "session_id": "IPRL/2023-06-30",
        "stored_frames": 7,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-e23cc600663294240959cca2",
        "session_id": "PennPAL/2023-04-29",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-e79bcde46bf467bcd3d0dbf1",
        "session_id": "AUTOLab/2023-10-21",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-e83ba5c884b8e685b5e1cbf2",
        "session_id": "TRI/2023-08-10",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f1e49ac52b58d2ae4a2dbbc3",
        "session_id": "RAIL/2023-07-15",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f3c3864e6e504a3a91eac459",
        "session_id": "RAIL/2023-10-04",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f57a4827f3ce0892c7a914f0",
        "session_id": "RAIL/2023-10-03",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f602c5d92e2d1d0f21386d02",
        "session_id": "RAIL/2023-10-26",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f65d3f2f4683f65eb47ca853",
        "session_id": "AUTOLab/2023-11-07",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f816aca98dad63575c1beb4b",
        "session_id": "AUTOLab/2023-12-18",
        "stored_frames": 6,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-f955caaebad88395e5c307e0",
        "session_id": "RAIL/2023-10-13",
        "stored_frames": 16,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-fdca52e69a5d87406bcba964",
        "session_id": "IRIS/2023-03-08",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      }
    ]
  },
  "validation_population": {
    "split": "val",
    "total_episodes": 143,
    "eligible_episodes": 132,
    "sessions": 57,
    "windows": 1351,
    "excluded_short": [
      {
        "episode_id": "droid-30384479bf99f329b60f8f23",
        "session_id": "RAIL/2023-11-29",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-34052e62aeee3c9f2384167f",
        "session_id": "RAIL/2023-09-21",
        "stored_frames": 20,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-508c771113f2fa326dd434ec",
        "session_id": "TRI/2023-08-22",
        "stored_frames": 7,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5c49e9f9cfb98d843fb76c76",
        "session_id": "REAL/2023-06-16",
        "stored_frames": 18,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-5c749a1534a6279d14526397",
        "session_id": "RAIL/2023-10-19",
        "stored_frames": 21,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-815b66b44d0a21fe0d6ba038",
        "session_id": "AUTOLab/2023-08-18",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-93f3b19cd2a2b43c64f2d372",
        "session_id": "REAL/2023-12-19",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-98679b224904071a1943259b",
        "session_id": "RAIL/2023-06-08",
        "stored_frames": 15,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-a83113f228f49cc895d3ad62",
        "session_id": "RAIL/2023-11-14",
        "stored_frames": 19,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-be80c3ffbadc6c389830b022",
        "session_id": "RAIL/2023-06-08",
        "stored_frames": 1,
        "reason": "fewer_than_23_stored_frames"
      },
      {
        "episode_id": "droid-bef6cdb91827b1e9bf514ffa",
        "session_id": "TRI/2023-08-09",
        "stored_frames": 22,
        "reason": "fewer_than_23_stored_frames"
      }
    ]
  },
  "hashes_verified_at_start_and_end": true,
  "gate_inference_cost": "2 donor ten-step prefix + 2 donor query rollouts; one-step control uses 20 extra one-step donor predictions during diagnostic collection"
}
