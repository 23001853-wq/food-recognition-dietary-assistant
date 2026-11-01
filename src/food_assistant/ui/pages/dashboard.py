from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from food_assistant.config import Settings
from food_assistant.ui.client import APIClient, disclaimer
from food_assistant.ui.labels import meal_label

MACROS = {"protein_g": "Protein (g)", "carbohydrate_g": "Carbohydrate (g)", "fat_g": "Chất béo (g)"}
COLORS = ["#2A9D8F", "#F4A261", "#567F71", "#A6CDBE"]


def chart_style(figure, height=300):
    figure.update_layout(
        height=height,
        margin=dict(l=12, r=12, t=12, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#264653", size=13),
        legend=dict(orientation="h", y=-0.2),
        hoverlabel=dict(bgcolor="#FFFFFF", font_color="#264653"),
    )
    figure.update_xaxes(showgrid=False, title=None, automargin=True)
    figure.update_yaxes(gridcolor="#E7F0EB", zeroline=False, rangemode="tozero", automargin=True)
    return figure


def render():
    st.title("Thống kê dinh dưỡng")
    st.caption("Nhìn lại thói quen ăn uống qua các bữa ăn đã ghi nhận.")
    today = datetime.now(ZoneInfo(Settings().app_timezone)).date()
    first, second = st.columns(2)
    start = first.date_input("Từ ngày", today - timedelta(days=6))
    end = second.date_input("Đến ngày", today)
    data = APIClient().request(
        "GET", "/api/v1/dashboard/summary", params={"start": str(start), "end": str(end)}
    )
    if data is None:
        return
    st.caption(
        f"{data['meal_count']} bữa ăn · {data['item_count']} món đã ghi nhận "
        f"trong khoảng {start:%d/%m} – {end:%d/%m/%Y}"
    )
    nutrients = data.get("nutrients", {})
    with st.container(key="dashboard_nutrients"):
        for column, field, label in zip(
            st.columns(4), ["estimated_kcal", *MACROS], ["Kcal ước tính", *MACROS.values()]
        ):
            values = nutrients.get(field, {})
            total = values.get("total")
            if field == "estimated_kcal" and not nutrients and data["known_count"]:
                total = data["estimated_kcal"]
            column.metric(label, f"{float(total):,.2f}" if total is not None else "Chưa có dữ liệu")
            if values:
                column.caption(
                    f"{values['known_count']} món có dữ liệu · "
                    f"{values['missing_count']} món chưa có"
                )
    st.caption(
        f"Tổng từ {data['known_count']} món có kcal; {data['missing_count']} món chưa xác định. "
        f"{data['user_input_count']} món dùng kcal người dùng nhập."
    )
    if data["daily"]:
        table = pd.DataFrame(data["daily"]).set_index("date")
        table.index = pd.to_datetime(table.index)
        table = table.reindex(pd.date_range(start, end))
        table["estimated_kcal"] = table["estimated_kcal"].astype(float)
        table.loc[table["known_count"] == 0, "estimated_kcal"] = float("nan")
        left, right = st.columns([1.7, 1])
        with left, st.container(border=True):
            st.subheader("Năng lượng theo ngày")
            fig = go.Figure(
                go.Scatter(
                    x=table.index,
                    y=table["estimated_kcal"],
                    mode="lines+markers",
                    line=dict(color=COLORS[0], width=3),
                    marker=dict(color=COLORS[1], size=8),
                    fill="tozeroy",
                    fillcolor="rgba(42,157,143,.09)",
                    connectgaps=False,
                    hovertemplate="%{x|%d/%m/%Y}<br>%{y:.2f} kcal<extra></extra>",
                )
            )
            fig.update_yaxes(title="Kcal ước tính")
            st.plotly_chart(chart_style(fig), width="stretch", config={"displayModeBar": False})
            st.caption("Khoảng trống là ngày chưa có kcal; tổng chỉ gồm các món có dữ liệu.")
        with right, st.container(border=True):
            st.subheader("Theo loại bữa ăn")
            known = {meal_label(k): float(v) for k, v in data["by_meal_type"].items()}
            if sum(known.values()) > 0:
                fig = go.Figure(
                    go.Pie(
                        labels=list(known),
                        values=list(known.values()),
                        hole=0.7,
                        marker=dict(colors=COLORS),
                        textinfo="percent",
                        sort=False,
                        hovertemplate="%{label}<br>%{value:.2f} kcal<extra></extra>",
                    )
                )
                st.plotly_chart(chart_style(fig), width="stretch", config={"displayModeBar": False})
                st.caption("Tỷ trọng kcal đã ghi, không phải tỷ lệ đạt mục tiêu dinh dưỡng.")
            else:
                st.info("Chưa có năng lượng lớn hơn 0 để thể hiện tỷ trọng.")
        with st.container(border=True):
            st.subheader("Các chất dinh dưỡng theo ngày")
            fig = go.Figure()
            for field, label, color in zip(MACROS, MACROS.values(), COLORS):
                values = [
                    day.get("nutrients", {}).get(field, {}).get("total") for day in data["daily"]
                ]
                fig.add_trace(
                    go.Bar(
                        name=label,
                        x=[d["date"] for d in data["daily"]],
                        y=[float(v) if v is not None else None for v in values],
                        marker_color=color,
                        hovertemplate="%{x}<br>%{y:.2f} g<extra>%{fullData.name}</extra>",
                    )
                )
            if any(nutrients.get(field, {}).get("known_count", 0) for field in MACROS):
                fig.update_layout(barmode="group")
                fig.update_yaxes(title="Gram đã ghi nhận")
                st.plotly_chart(chart_style(fig), width="stretch", config={"displayModeBar": False})
            else:
                st.info("Chưa có dữ liệu macro đã lưu trong khoảng ngày này.")
            st.caption(
                "Các chất được cộng độc lập từ snapshot. Không suy ra macro từ kcal nhập tay."
            )
        with st.expander("Xem số liệu theo ngày"):
            st.dataframe(
                table.drop(columns=["nutrients"], errors="ignore")
                .rename(
                    columns={
                        "estimated_kcal": "Kcal ước tính",
                        "known_count": "Món có kcal",
                        "missing_count": "Món chưa có kcal",
                    }
                )
                .rename_axis("Ngày"),
                width="stretch",
            )
        st.subheader("Các món thường gặp")
        foods = APIClient().request("GET", "/api/v1/foods") or []
        names = {f["id"]: f.get("display_name_vi") or f["display_name_en"] for f in foods}
        st.dataframe(
            [{"Món": names.get(k, k), "Số lần": v} for k, v in data["frequent_foods"].items()]
        )
    else:
        st.info("Chưa có dữ liệu nhật ký.")
    disclaimer()
