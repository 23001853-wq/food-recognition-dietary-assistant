import streamlit as st


def render_predictions(result):
    st.subheader("Kết quả dự đoán" + (" — MOCK" if result["is_mock"] else ""))
    st.dataframe(
        [
            {"Món": p["display_name_vi"], "Điểm xếp hạng": round(p["score"], 4)}
            for p in result["predictions"]
        ],
        hide_index=True,
    )
    st.caption("Score dùng để xếp hạng; chưa phải xác suất đã hiệu chuẩn.")
