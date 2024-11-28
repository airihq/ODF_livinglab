import streamlit as st
import pandas as pd
import numpy as np
import datetime


c_map = st.container()

# 공사 파일 경로
file_path = "./data/지오코딩_결과.csv"

# 공사 CSV 파일 읽기
site_df = pd.read_csv(file_path)
# '착공예정일', '실착공일', '사용승인일'에서 NaN 또는 빈 값이 있는 행을 제외
site_df = site_df.dropna(subset=["착공예정일", "실착공일", "사용승인일", "위도", "경도"])

site_df.rename(columns={"경도": "longitude", "위도": "latitude"}, inplace=True)

# '착공예정일'을 str로 변환하여 연도, 월을 추출하기 쉽게 함
site_df["착공예정일"] = site_df["착공예정일"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
site_df["실착공일"] = site_df["실착공일"].apply(lambda x: str(int(x)) if pd.notna(x) else "")
site_df["사용승인일"] = site_df["사용승인일"].apply(lambda x: str(int(x)) if pd.notna(x) else "")

# '착공예정일'에서 연도 및 월 추출
site_df["year"] = site_df["착공예정일"].apply(lambda x: int(x[:4]) if x else None)
site_df["month"] = site_df["착공예정일"].apply(lambda x: int(x[4:6]) if x else None)

# 사고지점 데이터
acci_df = pd.read_csv("./data/사고지점_통합_2007_2023.csv")

# 사고 범위 슬라이더의 값들
slider_values = [f"{year}-{month:02d}" for year in range(2007, 2024) for month in range(1, 13)]

# with c_slider:
# 사고 범위 슬라이더
start_date, end_date = st.sidebar.select_slider(
    label="Accident Date range",
    options=slider_values,
    value=("2023-01", "2023-12"),
)

# 슬라이더 결과의 년과 월 각각을 int로 변환
start_year, start_month = map(int, start_date.split("-"))
end_year, end_month = map(int, end_date.split("-"))

# 시간 범위에 해당하는 사고 DataFrame


def mask_date(df, start_year, start_month, end_year, end_month, exclusive=False):
    mask = (df["year"] > start_year) | ((df["year"] == start_year) & (df["month"] >= start_month))
    if exclusive:
        mask = mask & (
            (df["year"] < end_year) | ((df["year"] == end_year) & (df["month"] < end_month))
        )
    else:
        mask = mask & (
            (df["year"] < end_year) | ((df["year"] == end_year) & (df["month"] <= end_month))
        )
    return mask


isInRange_acci = mask_date(acci_df, start_year, start_month, end_year, end_month)

selected_acci_df = acci_df[isInRange_acci]

# label_mask = ["{화물차}" in l for l in selected_acci_df["labels"].to_numpy()]
# selected_acci_df = selected_acci_df[label_mask]

# 시간 범위에 해당하는 공사 DataFrame
isInRange_site = mask_date(site_df, start_year, start_month, end_year, end_month)

selected_site_df = site_df[isInRange_site]

# 둘의 마커 색깔과 크기 결정
selected_acci_df["color"] = [(0.0, 0, 0.8, 0.4)] * selected_acci_df.shape[0]
selected_acci_df["size"] = [60] * selected_acci_df.shape[0]

selected_site_df["color"] = [(1.0, 0.0, 0, 0.4)] * selected_site_df.shape[0]
selected_site_df["size"] = [120] * selected_site_df.shape[0]

# 출력할 데이터
res_df = pd.concat([selected_acci_df, selected_site_df])


with c_map:
    m = st.map(
        res_df,
        latitude="latitude",
        longitude="longitude",
        size="size",
        color="color",
        zoom=12,
        height=900,
    )
