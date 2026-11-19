# Food-101 verification report

- Overall: **FAIL**
- Mode: full
- Root: `C:\Users\User\MyProjects\food-recognition-dietary-assistant\data\raw\food-101`
- Images on disk: 101000
- Classes: 101
- Train/test entries: 75750 / 25250
- Valid / bad images: 101000 / 0
- Missing / unlisted images: 0 / 0
- Duplicate content groups: 8
- Train/test duplicate hashes: 2
- DataLoader status: NOT_RUN

## Warnings

- Exact duplicates may affect split independence and test evaluation. (8 duplicate groups; 2 cross-split hashes).

## Criteria

| Criterion | Status | Details |
|---|---|---|
| dataset_root | PASS | `"C:\\Users\\User\\MyProjects\\food-recognition-dietary-assistant\\data\\raw\\food-101"` |
| images_directory | PASS | `"data\\raw\\food-101\\images"` |
| metadata_directory | PASS | `"data\\raw\\food-101\\meta"` |
| required_metadata_files | PASS | `{"missing": []}` |
| class_labels_and_directories | PASS | `{"metadata_classes": 101, "unique_labels": 101, "image_directories": 101, "duplicate_labels": [], "directories_without_metadata": [], "metadata_without_directories": []}` |
| official_split_metadata | PASS | `{"train_entries": 75750, "test_entries": 25250, "train_per_class": {"apple_pie": 750, "baby_back_ribs": 750, "baklava": 750, "beef_carpaccio": 750, "beef_tartare": 750, "beet_salad": 750, "beignets": 750, "bibimbap": 750, "bread_pudding": 750, "breakfast_burrito": 750, "bruschetta": 750, "caesar_salad": 750, "cannoli": 750, "caprese_salad": 750, "carrot_cake": 750, "ceviche": 750, "cheesecake": 750, "cheese_plate": 750, "chicken_curry": 750, "chicken_quesadilla": 750, "chicken_wings": 750, "chocolate_cake": 750, "chocolate_mousse": 750, "churros": 750, "clam_chowder": 750, "club_sandwich": 750, "crab_cakes": 750, "creme_brulee": 750, "croque_madame": 750, "cup_cakes": 750, "deviled_eggs": 750, "donuts": 750, "dumplings": 750, "edamame": 750, "eggs_benedict": 750, "escargots": 750, "falafel": 750, "filet_mignon": 750, "fish_and_chips": 750, "foie_gras": 750, "french_fries": 750, "french_onion_soup": 750, "french_toast": 750, "fried_calamari": 750, "fried_rice": 750, "frozen_yogurt": 750` |
| official_json_split_metadata | PASS | `{"present": [true, true], "train_entries": 75750, "test_entries": 25250, "train_ids_match_text": true, "test_ids_match_text": true}` |
| official_display_labels | PASS | `{"label_count": 101, "class_count": 101}` |
| image_inventory | PASS | `{"actual_jpg": 101000, "expected_jpg": 101000, "per_class": {"apple_pie": 1000, "baby_back_ribs": 1000, "baklava": 1000, "beef_carpaccio": 1000, "beef_tartare": 1000, "beet_salad": 1000, "beignets": 1000, "bibimbap": 1000, "bread_pudding": 1000, "breakfast_burrito": 1000, "bruschetta": 1000, "caesar_salad": 1000, "cannoli": 1000, "caprese_salad": 1000, "carrot_cake": 1000, "ceviche": 1000, "cheesecake": 1000, "cheese_plate": 1000, "chicken_curry": 1000, "chicken_quesadilla": 1000, "chicken_wings": 1000, "chocolate_cake": 1000, "chocolate_mousse": 1000, "churros": 1000, "clam_chowder": 1000, "club_sandwich": 1000, "crab_cakes": 1000, "creme_brulee": 1000, "croque_madame": 1000, "cup_cakes": 1000, "deviled_eggs": 1000, "donuts": 1000, "dumplings": 1000, "edamame": 1000, "eggs_benedict": 1000, "escargots": 1000, "falafel": 1000, "filet_mignon": 1000, "fish_and_chips": 1000, "foie_gras": 1000, "french_fries": 1000, "french_onion_soup": 1000, "french_toast": 1000, "fried_calamari": 1000, "f` |
| license_file_present | PASS | `"data\\raw\\food-101\\license_agreement.txt"` |
| train_test_duplicate_content | FAIL | `{"matching_sha256_count": 2, "sha256": ["ec9b9f1437dde19e478d277d3519678d11f64435fbcf6f62ed965d048c9de9fe", "f40544646c0e07fcc3e6539bd03af71949dabb4b79515bfe207ca66e5f1c6f54"], "duplicate_path_groups": [["images/churros/2547960.jpg", "images/churros/2617186.jpg"], ["images/paella/2199939.jpg", "images/paella/2199941.jpg"]], "policy": "reject_cross_split_duplicates", "policy_version": 1, "policy_enabled": true}` |
| image_decoding_and_sha256 | PASS | `{"checked": 101000, "valid": 101000, "errors": 0, "bad_images": [], "duplicate_content_groups": 8, "duplicate_groups": [["images/churros/2547960.jpg", "images/churros/2617186.jpg"], ["images/dumplings/167000.jpg", "images/dumplings/180290.jpg"], ["images/hamburger/3578279.jpg", "images/hamburger/3783650.jpg"], ["images/omelette/2619753.jpg", "images/pancakes/2619752.jpg"], ["images/paella/2199939.jpg", "images/paella/2199941.jpg"], ["images/pho/4787.jpg", "images/pho/51112.jpg"], ["images/pho/51122.jpg", "images/pho/651427.jpg"], ["images/prime_rib/76271.jpg", "images/steak/97656.jpg"]], "train_test_duplicate_hashes": 2, "train_test_duplicate_sha256": ["ec9b9f1437dde19e478d277d3519678d11f64435fbcf6f62ed965d048c9de9fe", "f40544646c0e07fcc3e6539bd03af71949dabb4b79515bfe207ca66e5f1c6f54"]}` |
| download_completion_marker | PASS | `{"path": "data\\raw\\food-101\\download_metadata.json", "marker_claims": {"dataset": "food101", "academic_approval": "confirmed_by_user", "academic_approval_date": "2026-10-31", "source_url": "https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/", "archive_url": "https://data.vision.ee.ethz.ch/cvl/food-101.tar.gz", "use": "local academic research only; no redistribution of images", "rights_status": "Archive license inspected: scientific fair use; any other use requires picture-owner permission", "citation": "Bossard, Guillaumin, Van Gool. Food-101. ECCV 2014.", "official_sha256": null, "license_file": "food-101/license_agreement.txt", "license_verified_date": "2026-10-31", "download_sha256": "d97d15e438b7f4498f96086a4f7e2fa42a32f2712e87d3295441b2b6314053a4", "image_count": 101000, "official_checksum_verified": false}, "reason": "Legacy origin metadata is bound to the retained archive; v2 evidence was freshly recomputed.", "checksum": {"local_sha256": "d97d15e438b7f4498f96086a4f7` |

## Image issues

None reported.
