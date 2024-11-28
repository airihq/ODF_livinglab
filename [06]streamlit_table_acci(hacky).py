import streamlit as st
import pandas as pd

# 첨부한 파일 경로
file_path = "./data/사고지점_통합_2007_2023.csv"

# CSV 파일 읽기
df = pd.read_csv(file_path)

# 사이드바에 select box를 활용하여 동을 선택.
st.sidebar.title("광명시 교통사고 데이터")

select_multi_species = st.sidebar.multiselect(
    "사고유형 (복수선택가능)",
    [
        "차대사람",
        "승합차",
        "이륜차",
        "택시",
        "신호위반",
        "화물차",
        "렌터카",
        "승용차",
        "차량단독",
        "시내버스",
        "고령운전자",
        "자전거",
        "원동기장치자전거",
        "음주운전",
        "사업용자동차",
    ],
)

# '착공예정일', '실착공일', '사용승인일'에서 NaN 또는 빈 값이 있는 행을 제외
# df = df.dropna(subset=['착공예정일', '실착공일', '사용승인일'])

# '착공예정일'을 str로 변환하여 연도, 월을 추출하기 쉽게 함
# df['착공예정일'] = df['착공예정일'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
# df['실착공일'] = df['실착공일'].apply(lambda x: str(int(x)) if pd.notna(x) else '')
# df['사용승인일'] = df['사용승인일'].apply(lambda x: str(int(x)) if pd.notna(x) else '')

# '착공예정일'에서 연도 및 월 추출
# df['연도'] = df['착공예정일'].apply(lambda x: x[:4] if x else None)
# df['월'] = df['착공예정일'].apply(lambda x: x[4:6] if x else None)

# Streamlit 슬라이더로 연도와 월 범위 선택
min_year = int(df["year"].min())
max_year = int(df["year"].max())

# 연도 범위 슬라이더 (최소 연도와 최대 연도 사이)
selected_year_range = st.sidebar.slider(
    "연도 범위 선택", min_value=min_year, max_value=max_year, value=(min_year, max_year)
)

# 월 범위 슬라이더 (1월부터 12월까지)
selected_month_range = st.sidebar.slider("월 범위 선택", min_value=1, max_value=12, value=(1, 12))

# 선택한 연도 범위 및 월 범위로 데이터 필터링
start_year, end_year = selected_year_range
start_month, end_month = selected_month_range


def mask_date(df, start_year, start_month, end_year, end_month):
    mask = (
        (df["year"] > start_year) | ((df["year"] == start_year) & (df["month"] >= start_month))
    ) & (
        (df["year"] < end_year) | ((df["year"] == end_year) & (df["month"] <= end_month))
    )  # **** inclusive ****
    return mask


filtered_df = df[mask_date(df, start_year, start_month, end_year, end_month)]


filtered_df["year"] = filtered_df["year"].apply(lambda x: f"{x:d}")
filtered_df["month"] = filtered_df["month"].astype(str)
filtered_df.rename(
    columns={
        "year": "연도",
        "month": "월",
        "utmk_x": "UTM-K x좌표",
        "utmk_y": "UTM-K y좌표",
        "longitude": "경도",
        "latitude": "위도",
        "labels": "사고유형",
    },
    inplace=True,
)


# 필터링된 데이터 테이블 출력
st.write(f"선택된 연도 범위: {start_year} - {end_year}, 월 범위: {start_month} - {end_month}")
st.write(filtered_df)
