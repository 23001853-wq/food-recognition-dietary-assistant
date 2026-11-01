import streamlit as st


def render_estimate(estimate):
    st.subheader("🥗 Dinh dưỡng ước tính")
    with st.container(key="nutrition_estimate"):
        columns = st.columns(4)
        for column, key, label in zip(
            columns,
            ["estimated_kcal", "protein_g", "carbohydrate_g", "fat_g"],
            ["Kcal ước tính", "Protein (g)", "Carbohydrate (g)", "Chất béo (g)"],
        ):
            column.metric(
                label, estimate.get(key) if estimate.get(key) is not None else "Chưa có dữ liệu"
            )
    if estimate.get("source_type") == "user_input":
        st.caption("Nguồn: Người dùng nhập (user_input). Không thay đổi dữ liệu USDA.")
    elif estimate.get("source_name"):
        st.caption(
            f"Nguồn: {estimate['source_name']} · FDC {estimate['source_record_id']} · "
            f"{estimate['source_version']}"
        )
        st.caption(
            f"Thực phẩm tham chiếu: {estimate['reference_name']} · "
            f"{estimate.get('kcal_per_100g')} kcal/100 g"
        )
        st.caption(estimate.get("source") or "")
    if estimate.get("grams") is not None:
        st.info(f"Khối lượng: {estimate['grams']} g")
    for key in ("portion_note", "match_note"):
        if estimate.get(key):
            st.caption(estimate[key])
    if estimate.get("status") == "unavailable":
        st.info(
            "Chưa có đủ dữ liệu dinh dưỡng hoặc gram để tính kcal. "
            "Giá trị thiếu được giữ là chưa xác định."
        )
