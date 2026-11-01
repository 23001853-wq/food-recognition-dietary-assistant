# Nutrition assumptions

Current catalog covers the exact ten POC classes, using seven USDA SR Legacy records
and three USDA FNDDS 2021–2023 records. See [NUTRITION_GUIDE.md](../NUTRITION_GUIDE.md)
for record IDs, amounts, caveats, update commands and verification steps.

Every Small/Medium/Large mass is an explicit project design assumption, not an official
USDA serving or a vision measurement. All energy and macros come directly from source
nutrient records per 100 g. Representative mappings are not recipe-specific ground truth:
Caesar salad is explicitly without dressing, ice cream is vanilla, curry excludes rice,
and generic sushi cannot cover every filling. Missing information remains NULL.

Manual energy is tagged `user_input`, with no invented macros. Meal snapshots retain
reference identity/version, source values and mass provenance. Import preserves old
profiles, default mappings and diaries. The original pizza profile and source provenance
remain intact; full current coverage and computed archive hashes are in
`data/nutrition/coverage.json`. The older `provenance.json` documents pizza's initial import.

This estimates nutrition from a selected or entered mass; it does not measure a food's
mass or composition from image pixels and is not personalized medical advice.
