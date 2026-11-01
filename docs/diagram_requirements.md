# Diagram requirements

Architecture v2: Browser → Streamlit → FastAPI → services → PostgreSQL / trusted model artifacts. Sources in diagrams/ reflect this boundary; the ERD is generated from current ORM by scripts/update_diagrams.py. Diagrams are explanatory designs, not evidence of trained-model performance. Future clients/auth/detection remain explicitly Future. No image/weight dataset is stored in PostgreSQL. Prediction metadata and meal snapshots preserve provenance. Mermaid sources have not been rendered by a Mermaid runtime in this session; no SVG/PNG output is claimed.
