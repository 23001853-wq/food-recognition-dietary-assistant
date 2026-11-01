"""HTTP-only UI boundary; never imports persistence or training."""

import httpx
import streamlit as st

from food_assistant.config import Settings


class APIClient:
    def __init__(self):
        self.base = Settings().api_base_url.rstrip("/")

    def request(self, method, path, **kwargs):
        try:
            response = httpx.request(method, self.base + path, timeout=60, **kwargs)
            if response.is_error:
                try:
                    detail = response.json()
                except ValueError:
                    detail = {"error": "Backend returned an invalid response"}
                st.error(f"API {response.status_code}: {detail}")
                return None
            return response.json() if response.content else {"deleted": True}
        except httpx.HTTPError:
            st.error("Không kết nối được backend. Kiểm tra FastAPI và API_BASE_URL.")
            return None


def disclaimer():
    st.caption(
        "Giá trị dinh dưỡng chỉ là ước tính theo dữ liệu tham khảo, không phải đo trực tiếp từ ảnh"
    )


def food_label(food):
    return food.get("display_name_vi") or food["display_name_en"]
