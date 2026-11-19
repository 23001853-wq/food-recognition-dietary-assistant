# Phase 4 — Full Validation on cleaned split v3

- Run ID: `full-v3-20261118T125620Z-77f1738f`
- Timestamp UTC: `2026-11-18T12:56:20.911048+00:00`
- Commit / dirty: `2ead0db` / `True`
- Scope requested / actual: full Food-101 v3 / 101 classes, 100990 rows
- Status: **PASS**
- Input split: `C:\Users\User\MyProjects\food-recognition-dietary-assistant\data\splits\v3` (v3)

## Counts and reconciliation

- Train / validation / test: 68166 / 7574 / 25250
- Classes: 101 (expected 101)
- Official train IDs removed by v2: 2
- Additional v3 train exclusions: 8
- Official test retained: 25250; order/IDs match: True
- Exclusion ledger entries: 8
- Official train retained in v3: 75740; test retained: 25250
- Same-label groups removed: 4; cross-label groups quarantined: 2

## Quality and evidence

- Unique IDs: 100990; missing: 0
- Exact cross-partition hash groups: 0
- Same-split exact duplicate groups: 0
- Capture groups: 0; cross-partition: 0
- Source fingerprint: `f5cfdb7480c2650ed94c8624652ac595e4a7f038d660f4886869836632c56ed1`
- Inventory SHA-256: `b91c327114b45c4fe2e626404a86c449891e6e258c832e67a7f2350aeb59adb9`
- v2 fingerprint: `319780d638052e9bfddbf28aa9e3df9bda9d9d78224a220ef7603b3d02bb150e`
- v3 fingerprint: `aa43cfa766982006fe6a3611a49e0f07d35d33d56e54310e923cb5e58fbdbb59`
- Evidence reuse: Phase 2.5 cached full decode/SHA results reused after current file signatures and source fingerprints matched.
- Near duplicate audit: NOT_CHECKED; evidence and policy cover exact SHA-256 duplicates only
- Test isolation: PASS by v3 policy and implementation review; test IDs/order/checksums are preserved from v2 and validation derives only from train candidates.
- Raw checkpoint: `data\reports\elt\v2\phase_2_5_verification_retry\food101_checkpoint.json`
- v3 metadata/report: `C:\Users\User\MyProjects\food-recognition-dietary-assistant\data\splits\v3\dataset_metadata.json` / `C:\Users\User\MyProjects\food-recognition-dietary-assistant\data\splits\v3\quality_report.json`

## Warnings

- Historical Phase 2.5 official train/test leakage FAIL is retained; it is not applied to the cleaned v3 partition.
- No near-duplicate or semantic duplicate audit is claimed.
- EXIF/TIFF warning count is unavailable in historical checkpoint evidence.

## Errors

None.

## Per-class distribution

| Class | Train | Validation | Test |
|---|---:|---:|---:|
| apple_pie | 675 | 75 | 250 |
| baby_back_ribs | 675 | 75 | 250 |
| baklava | 675 | 75 | 250 |
| beef_carpaccio | 675 | 75 | 250 |
| beef_tartare | 675 | 75 | 250 |
| beet_salad | 675 | 75 | 250 |
| beignets | 675 | 75 | 250 |
| bibimbap | 675 | 75 | 250 |
| bread_pudding | 675 | 75 | 250 |
| breakfast_burrito | 675 | 75 | 250 |
| bruschetta | 675 | 75 | 250 |
| caesar_salad | 675 | 75 | 250 |
| cannoli | 675 | 75 | 250 |
| caprese_salad | 675 | 75 | 250 |
| carrot_cake | 675 | 75 | 250 |
| ceviche | 675 | 75 | 250 |
| cheesecake | 675 | 75 | 250 |
| cheese_plate | 675 | 75 | 250 |
| chicken_curry | 675 | 75 | 250 |
| chicken_quesadilla | 675 | 75 | 250 |
| chicken_wings | 675 | 75 | 250 |
| chocolate_cake | 675 | 75 | 250 |
| chocolate_mousse | 675 | 75 | 250 |
| churros | 674 | 75 | 250 |
| clam_chowder | 675 | 75 | 250 |
| club_sandwich | 675 | 75 | 250 |
| crab_cakes | 675 | 75 | 250 |
| creme_brulee | 675 | 75 | 250 |
| croque_madame | 675 | 75 | 250 |
| cup_cakes | 675 | 75 | 250 |
| deviled_eggs | 675 | 75 | 250 |
| donuts | 675 | 75 | 250 |
| dumplings | 674 | 75 | 250 |
| edamame | 675 | 75 | 250 |
| eggs_benedict | 675 | 75 | 250 |
| escargots | 675 | 75 | 250 |
| falafel | 675 | 75 | 250 |
| filet_mignon | 675 | 75 | 250 |
| fish_and_chips | 675 | 75 | 250 |
| foie_gras | 675 | 75 | 250 |
| french_fries | 675 | 75 | 250 |
| french_onion_soup | 675 | 75 | 250 |
| french_toast | 675 | 75 | 250 |
| fried_calamari | 675 | 75 | 250 |
| fried_rice | 675 | 75 | 250 |
| frozen_yogurt | 675 | 75 | 250 |
| garlic_bread | 675 | 75 | 250 |
| gnocchi | 675 | 75 | 250 |
| greek_salad | 675 | 75 | 250 |
| grilled_cheese_sandwich | 675 | 75 | 250 |
| grilled_salmon | 675 | 75 | 250 |
| guacamole | 675 | 75 | 250 |
| gyoza | 675 | 75 | 250 |
| hamburger | 674 | 75 | 250 |
| hot_and_sour_soup | 675 | 75 | 250 |
| hot_dog | 675 | 75 | 250 |
| huevos_rancheros | 675 | 75 | 250 |
| hummus | 675 | 75 | 250 |
| ice_cream | 675 | 75 | 250 |
| lasagna | 675 | 75 | 250 |
| lobster_bisque | 675 | 75 | 250 |
| lobster_roll_sandwich | 675 | 75 | 250 |
| macaroni_and_cheese | 675 | 75 | 250 |
| macarons | 675 | 75 | 250 |
| miso_soup | 675 | 75 | 250 |
| mussels | 675 | 75 | 250 |
| nachos | 675 | 75 | 250 |
| omelette | 674 | 75 | 250 |
| onion_rings | 675 | 75 | 250 |
| oysters | 675 | 75 | 250 |
| pad_thai | 675 | 75 | 250 |
| paella | 674 | 75 | 250 |
| pancakes | 674 | 75 | 250 |
| panna_cotta | 675 | 75 | 250 |
| peking_duck | 675 | 75 | 250 |
| pho | 674 | 74 | 250 |
| pizza | 675 | 75 | 250 |
| pork_chop | 675 | 75 | 250 |
| poutine | 675 | 75 | 250 |
| prime_rib | 674 | 75 | 250 |
| pulled_pork_sandwich | 675 | 75 | 250 |
| ramen | 675 | 75 | 250 |
| ravioli | 675 | 75 | 250 |
| red_velvet_cake | 675 | 75 | 250 |
| risotto | 675 | 75 | 250 |
| samosa | 675 | 75 | 250 |
| sashimi | 675 | 75 | 250 |
| scallops | 675 | 75 | 250 |
| seaweed_salad | 675 | 75 | 250 |
| shrimp_and_grits | 675 | 75 | 250 |
| spaghetti_bolognese | 675 | 75 | 250 |
| spaghetti_carbonara | 675 | 75 | 250 |
| spring_rolls | 675 | 75 | 250 |
| steak | 674 | 75 | 250 |
| strawberry_shortcake | 675 | 75 | 250 |
| sushi | 675 | 75 | 250 |
| tacos | 675 | 75 | 250 |
| takoyaki | 675 | 75 | 250 |
| tiramisu | 675 | 75 | 250 |
| tuna_tartare | 675 | 75 | 250 |
| waffles | 675 | 75 | 250 |

## Cleanup provenance

| Image ID | Class | Reason | Representative/matches |
|---|---|---|---|
| dumplings/180290 | dumplings | same_label_exact_duplicate_deduplicated | dumplings/167000 |
| hamburger/3783650 | hamburger | same_label_exact_duplicate_deduplicated | hamburger/3578279 |
| omelette/2619753 | omelette | cross_label_exact_duplicate_quarantined | pancakes/2619752 |
| pancakes/2619752 | pancakes | cross_label_exact_duplicate_quarantined | omelette/2619753 |
| pho/51112 | pho | same_label_exact_duplicate_deduplicated | pho/4787 |
| pho/651427 | pho | same_label_exact_duplicate_deduplicated | pho/51122 |
| prime_rib/76271 | prime_rib | cross_label_exact_duplicate_quarantined | steak/97656 |
| steak/97656 | steak | cross_label_exact_duplicate_quarantined | prime_rib/76271 |
