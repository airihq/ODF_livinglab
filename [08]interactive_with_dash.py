import json
import datetime
import base64
from io import BytesIO

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    callback,
    dash_table,
    ctx,
    no_update,
    Patch,
)
from dash.exceptions import PreventUpdate
from dash.dash_table.Format import Format, Scheme
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np
from pyproj import Transformer
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib import patches
import cv2

matplotlib.use("agg")

font_scale = 1.2

font_hangeul = font_manager.FontProperties(
    fname="./fonts/Pretendard-Regular.ttf", size=20 * font_scale
)
font_hangeul_legend = font_manager.FontProperties(
    fname="./fonts/Pretendard-Regular.ttf", size=12 * font_scale
)
font_hangeul_legend2 = font_manager.FontProperties(
    fname="./fonts/Pretendard-Regular.ttf", size=14 * font_scale
)


acci_df = pd.read_csv("./data/사고지점_통합_2007_2023.csv")

raw_acci_columns = [{"name": i, "id": i} for i in acci_df.columns]
# utmk
for i in range(4, 6):
    raw_acci_columns[i]["type"] = "numeric"
    raw_acci_columns[i]["format"] = Format(precision=2, scheme=Scheme.fixed)
# 위도 경도
for i in range(6, 8):
    raw_acci_columns[i]["type"] = "numeric"
    raw_acci_columns[i]["format"] = Format(precision=4, scheme=Scheme.fixed)


site_df = pd.read_csv("./data/지오코딩_결과.csv")

raw_site_columns = [{"name": i, "id": i} for i in site_df.columns]

# 위도 경도
for i in range(12, 14):
    raw_site_columns[i]["type"] = "numeric"
    raw_site_columns[i]["format"] = Format(precision=4, scheme=Scheme.fixed)


site_df = site_df[site_df["위도"] > 37.39]
# 전처리
# 경도위도 => utm-k 좌표로 변환

gcs_to_utmk = Transformer.from_crs("EPSG:4326", "EPSG:5178", always_xy=True)

tmp_x, tmp_y = gcs_to_utmk.transform(site_df["경도"], site_df["위도"])
site_df["utmk_x"] = tmp_x
site_df["utmk_y"] = tmp_y

site_date = site_df["실착공일"].copy()

site_date.loc[site_df["실착공일"].isna()] = site_df.loc[site_df["실착공일"].isna(), "착공예정일"]

site_date = site_date.apply(lambda x: str(int(x)))

site_df["year"], site_df["month"] = site_date.apply(lambda x: int(x[:4])), site_date.apply(
    lambda x: int(x[4:6])
)

time_ticks = [f"{year}-{month:02d}" for year in range(2007, 2023) for month in range(1, 13)]

# 나중에 customdata로 저장하여 접근하기 쉽도록 인덱스 저장
site_df.insert(0, "index", list(range(len(site_df))))


def get_end_dates(df):
    end_date = df["사용승인일"].copy(deep=True)
    end_date[pd.isna(end_date)] = df["실착공일"][pd.isna(end_date)]
    end_date[pd.isna(end_date)] = df["착공예정일"][pd.isna(end_date)]
    end_date = end_date.apply(lambda x: str(int(x)))
    year, month = end_date.apply(lambda x: int(x[:4])), end_date.apply(lambda x: int(x[4:6]))
    return year, month


def mask_date(tgt_year, tgt_month, start_year, start_month, end_year, end_month, exclusive=False):
    mask = (tgt_year > start_year) | ((tgt_year == start_year) & (tgt_month >= start_month))
    if exclusive:
        mask = mask & ((tgt_year < end_year) | ((tgt_year == end_year) & (tgt_month < end_month)))
    else:
        mask = mask & (
            (tgt_year < end_year) | ((tgt_year == end_year) & (tgt_month <= end_month))
        )
    return mask


def mask_start_date(df, start_year, start_month, end_year, end_month, exclusive=False):
    return mask_date(
        df["year"], df["month"], start_year, start_month, end_year, end_month, exclusive
    )


def mask_end_date(df, start_year, start_month, end_year, end_month, exclusive=False):
    df_year, df_month = get_end_dates(df)
    return mask_date(df_year, df_month, start_year, start_month, end_year, end_month, exclusive)


def get_df_in_range(start, end):
    start_year, start_month = [int(i) for i in start.split("-")]
    end_year, end_month = [int(i) for i in end.split("-")]
    site_dff = site_df[
        mask_start_date(site_df, start_year, start_month, end_year, end_month)
    ].copy(deep=True)
    acci_dff = acci_df[
        mask_start_date(acci_df, start_year, start_month, end_year, end_month)
    ].copy(deep=True)
    return site_dff, acci_dff


def draw_mapfig(site_dff, acci_dff, initial=True):
    if not hasattr(draw_mapfig, "initial_map"):
        initial = True
    site_dff["color"] = np.array(["공사위치" for _ in range(len(site_dff))])
    acci_dff["color"] = np.array(["교통사고" for _ in range(len(acci_dff))])
    acci_dff["사고년월"] = np.array(
        [f"{year}-{month:02d}" for year, month in acci_dff.loc[:, ["year", "month"]].to_numpy()]
    )
    if initial:

        fig = make_subplots()

        site_fig = px.scatter_map(
            site_dff,
            lat="위도",
            lon="경도",
            color="color",
            color_discrete_map={"공사위치": "#1111DF"},
            color_continuous_scale=px.colors.cyclical.IceFire,
            custom_data=["index", "실착공일", "사용승인일"],
            hover_name="대지위치",
            hover_data={"실착공일": True, "사용승인일": True, "연면적(㎡)": True, "color": False},
            size_max=15,
            zoom=12,
        )
        fig.add_traces(site_fig.data)
        # if not initial:
        #     print("lol")
        #     # site_fig.layout["map"] = None
        #     # site_fig.layout["mapbox"] = None

        #     site_fig.layout["map"] = draw_mapfig.initial_map
        #     site_fig.layout["mapbox"] = draw_mapfig.initial_mapbox
        # fig.update_layout(site_fig.layout)

        acci_fig = px.scatter_map(
            acci_dff,
            lat="latitude",
            lon="longitude",
            color="color",
            color_discrete_map={"교통사고": "#E70000"},
            size_max=15,
            hover_data={
                "사고년월": ":.s",
                "latitude": ":.4f",
                "longitude": ":.4f",
                "labels": ":.s",
                "color": False,
            },
            labels={"latitude": "위도", "longitude": "경도", "labels": "사고유형"},
            opacity=0.2,
        )
        # acci_fig.update_traces(hovertemplate=None)
        fig.add_traces(acci_fig.data)
        if initial:
            fig.update_layout(site_fig.layout)
            draw_mapfig.initial_mapbox = site_fig.layout["mapbox"]
            draw_mapfig.initial_map = site_fig.layout["map"]
        print(fig)
        return fig
    else:
        # raise PreventUpdate
        fig = Patch()
        fig["data"][0]["lat"] = site_dff["위도"].to_numpy()
        fig["data"][0]["lon"] = site_dff["경도"].to_numpy()
        # fig["data"][0]["marker"]["color"] = site_dff["color"].to_numpy()
        fig["data"][0]["customdata"] = site_dff.loc[
            :, ["index", "실착공일", "사용승인일"]
        ].to_numpy()
        fig["data"][0]["hovertext"] = site_dff["대지위치"].to_numpy()

        fig["data"][1]["lat"] = acci_dff["latitude"].to_numpy()
        fig["data"][1]["lon"] = acci_dff["longitude"].to_numpy()
        # fig["data"][1]["marker"]["color"] = acci_dff["color"].to_numpy()
        fig["data"][1]["customdata"] = acci_dff.loc[
            :, ["사고년월", "latitude", "longitude", "labels"]
        ].to_numpy()
        return fig


def get_acci_range(site_year, site_month, timespan_month=6):
    # return start (year, month), end (year, month) (for exclusive range)
    if timespan_month <= 0:
        end_year = site_year + site_month // 12
        end_month = site_month % 12 + 1
        return site_year, site_month, end_year, end_month
    start_year = site_year + (site_month - timespan_month - 1) // 12
    start_month = (site_month - timespan_month - 1) % 12 + 1
    end_year = site_year + (site_month + timespan_month - 1) // 12
    end_month = (site_month + timespan_month - 1) % 12 + 1
    return start_year, start_month, end_year, end_month


def mask_distance(df, x, y, d=1000):
    mask = (df["utmk_x"] - x) ** 2 + (df["utmk_y"] - y) ** 2 <= d**2
    return mask


# 공사 총 개월수 계산 (준공일이 없다면 1개월로 취급)
def get_site_duration(site, start_year=None, start_month=None):
    if (start_year is None) or (start_month is None):
        start_year, start_month = site["year"], site["month"]
    site_duration = 1
    if not pd.isna(site["사용승인일"]):
        site_end_year = int(f"{site['사용승인일']:.0f}"[:4])
        site_end_month = int(f"{site['사용승인일']:.0f}"[4:6])
        site_duration = max(
            1, (site_end_year - start_year) * 12 + (site_end_month - start_month) + 1
        )
    return site_duration


app = Dash(__name__)

graph_comp = dcc.Graph(
    id="graph-map",
    style={"height": "900px"},
    figure=draw_mapfig(site_df, acci_df),
)

initial_site_img = plt.figure(num=1, figsize=(13, 17), clear=True)
initial_site_img.text(
    0.5,
    0.5,
    "왼쪽에서 공사 지점 클릭 시 분석 이미지를 그립니다.",
    ha="center",
    va="center",
    font=font_hangeul,
)
buf = BytesIO()
initial_site_img.savefig(buf, format="png")

initial_site_img = base64.b64encode(buf.getbuffer()).decode("ascii")
initial_site_img = f"data:image/png;base64,{initial_site_img}"

initial_site_dff, initial_acci_dff = get_df_in_range(time_ticks[0], time_ticks[-1])

upper_left_panel = html.Div(
    [
        html.Div(
            [
                html.Div(
                    children=[
                        html.Div(
                            children="시작 년도",
                            style={
                                "text-align": "right",
                                "position": "relative",
                                "top": "50%",
                                "transform": "translate(0, -50%)",
                            },
                        )
                    ],
                    style={"position": "relative"},
                ),
                dcc.Dropdown(
                    [i for i in range(2007, 2023)],
                    value=2007,
                    id="start-year",
                    clearable=False,
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="끝 년도",
                            style={
                                "text-align": "right",
                                "position": "relative",
                                "top": "50%",
                                "transform": "translate(0, -50%)",
                            },
                        )
                    ],
                    style={"position": "relative"},
                ),
                dcc.Dropdown(
                    [i for i in range(2007, 2023)],
                    value=2022,
                    id="end-year",
                    clearable=False,
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="시작 월",
                            style={
                                "text-align": "right",
                                "position": "relative",
                                "top": "50%",
                                "transform": "translate(0, -50%)",
                            },
                        )
                    ],
                    style={"position": "relative"},
                ),
                dcc.Dropdown(
                    [i for i in range(1, 13)],
                    value=1,
                    id="start-month",
                    clearable=False,
                ),
                html.Div(
                    children=[
                        html.Div(
                            children="끝 월",
                            style={
                                "text-align": "right",
                                "position": "relative",
                                "top": "50%",
                                "transform": "translate(0, -50%)",
                            },
                        )
                    ],
                    style={"position": "relative"},
                ),
                dcc.Dropdown(
                    [i for i in range(1, 13)],
                    value=12,
                    id="end-month",
                    clearable=False,
                ),
            ],
            style={
                "display": "grid",
                "grid-template-columns": "1fr 2fr 1fr 2fr",
                "grid-template-rows": "auto auto",
                "padding": "10px",
                "gap": "20px",
            },
        ),
        dcc.RangeSlider(
            min=0,
            max=len(time_ticks) - 1,
            value=[0, len(time_ticks) - 1],
            step=1,
            marks={i: time_ticks[i][:4] for i in range(0, len(time_ticks), 12)},
            pushable=0,
            id="time-range",
        ),
    ],
    style={
        "height": "175px",
    },
)

upper_right_panel = html.Div(
    [
        html.Div(
            children=[
                html.Div(
                    children="시간 범위(개월)",
                    style={
                        "text-align": "right",
                        "position": "relative",
                        "top": "50%",
                        "transform": "translate(0, -50%)",
                    },
                )
            ],
            style={"position": "relative"},
        ),
        dcc.Input(
            value="12",
            type="number",
            placeholder="분석할 n개월을 입력하십시오...",
            id="input-time-span",
            debounce=True,
        ),
        html.Div(
            children=[
                html.Div(
                    children="이웃 반경",
                    style={
                        "text-align": "right",
                        "position": "relative",
                        "top": "50%",
                        "transform": "translate(0, -50%)",
                    },
                )
            ],
            style={"position": "relative"},
        ),
        dcc.Input(
            value="1000",
            type="number",
            placeholder="분석할 이웃 반경을 입력하십시오...",
            id="input-radius",
            debounce=True,
        ),
        html.Div(
            children=[
                html.Div(
                    children="연면적(㎡)이상",
                    style={
                        "text-align": "right",
                        "position": "relative",
                        "top": "50%",
                        "transform": "translate(0, -50%)",
                    },
                )
            ],
            style={"position": "relative"},
        ),
        dcc.Input(
            value=f"{max(int(site_df['연면적(㎡)'].min())-1, 0)}",
            type="number",
            placeholder="기준 연면적을 입력하십시오...",
            id="input-gross-floor",
            debounce=True,
        ),
        html.Div(
            children=html.Button(
                children="공사 분석 수동 갱신",
                id="site-graph-apply",
                style={
                    "width": "50%",
                    "height": "100%",
                },
            ),
            style={
                "grid-column": "3 / 5",
                "text-align": "center",
                "align-content": "center",
            },
        ),
    ],
    style={
        "display": "grid",
        "grid-template-columns": "3fr 5fr 3fr 5fr",
        "grid-template-rows": "auto auto",
        "padding": "10px",
        "gap": "20px",
        "height": "100px",
    },
)


app.layout = html.Div(
    [
        html.Div(
            [
                # left elements,
                upper_left_panel,
                graph_comp,
            ],
            style={"height": "100vh"},
        ),
        html.Div(
            [
                # right elements
                dcc.Tabs(
                    [
                        dcc.Tab(
                            [
                                html.Div(
                                    [
                                        html.H2("설정한 시간 범위 내 공사 데이터"),
                                        dash_table.DataTable(
                                            initial_site_dff.to_dict("records"),
                                            raw_site_columns,
                                            page_size=5,
                                            style_data={"whiteSpace": "normal", "height": "auto"},
                                            style_table={"overflowX": "auto"},
                                            id="site-table",
                                        ),
                                        html.H2("설정한 시간 범위 내 사고 데이터"),
                                        dash_table.DataTable(
                                            initial_acci_dff.to_dict("records"),
                                            raw_acci_columns,
                                            page_size=10,
                                            style_data={"whiteSpace": "normal", "height": "auto"},
                                            style_table={"overflowX": "auto"},
                                            id="acci-table",
                                        ),
                                    ],
                                    style={"width": "100%"},
                                ),
                            ],
                            label="공사/사고 데이터 표",
                            value="table-tab",
                            style={"align-content": "center", "padding": "0px"},
                            selected_style={"align-content": "center", "padding": "0px"},
                        ),
                        dcc.Tab(
                            [
                                upper_right_panel,
                                html.Img(
                                    src=initial_site_img, id="tgt-site-graph", height="850px"
                                ),
                            ],
                            label="공사 분석",
                            value="site-graph-tab",
                            style={"align-content": "center", "padding": "0px"},
                            selected_style={"align-content": "center", "padding": "0px"},
                        ),
                    ],
                    id="right-tabs",
                    value="table-tab",
                    style={"height": "44px"},
                )
            ]
        ),
    ],
    style={
        "display": "grid",
        "grid-template-columns": "50% 50%",
    },
)


@callback(
    Output("start-year", "value"),
    Output("start-month", "value"),
    Output("end-year", "value"),
    Output("end-month", "value"),
    Output("time-range", "value"),
    Input("start-year", "value"),
    Input("start-month", "value"),
    Input("end-year", "value"),
    Input("end-month", "value"),
    Input("time-range", "value"),
)
def validate_range(st_year, st_month, end_year, end_month, time_range):
    # print(ctx.triggered)
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if "start" in triggered_id:
        if (st_year > end_year) or (st_year == end_year and st_month > end_month):
            return (
                no_update,
                no_update,
                st_year,
                st_month,
                [
                    time_ticks.index(f"{st_year}-{st_month:02d}"),
                    time_ticks.index(f"{st_year}-{st_month:02d}"),
                ],
            )
    elif "end" in triggered_id:
        if (st_year > end_year) or (st_year == end_year and st_month > end_month):
            return (
                end_year,
                end_month,
                no_update,
                no_update,
                [
                    time_ticks.index(f"{end_year}-{end_month:02d}"),
                    time_ticks.index(f"{end_year}-{end_month:02d}"),
                ],
            )
    elif "range" in triggered_id:
        st, end = time_ticks[time_range[0]], time_ticks[time_range[1]]
        st_year, st_month = (int(c) for c in st.split("-"))
        end_year, end_month = (int(c) for c in end.split("-"))
        return st_year, st_month, end_year, end_month, no_update
    return (
        no_update,
        no_update,
        no_update,
        no_update,
        [
            time_ticks.index(f"{st_year}-{st_month:02d}"),
            time_ticks.index(f"{end_year}-{end_month:02d}"),
        ],
    )


@callback(
    Output("graph-map", "figure"),
    Output("site-table", "data"),
    Output("acci-table", "data"),
    Input("start-year", "value"),
    Input("start-month", "value"),
    Input("end-year", "value"),
    Input("end-month", "value"),
    # Input("time-range", "value"),
)
def update_map(st_year, st_month, end_year, end_month):
    site_dff, acci_dff = get_df_in_range(
        f"{st_year}-{st_month:02d}", f"{end_year}-{end_month:02d}"
    )
    fig = draw_mapfig(site_dff, acci_dff, initial=False)
    # fig.update_layout(uirevision=1)
    # fig.update_xaxes(autorange=False)
    # fig.update_yaxes(autorange=False)
    return fig, site_dff.to_dict("records"), acci_dff.to_dict("records")


# 그림용 변수
figpad = 500
dot_scale = 15

bg_img = plt.imread("./data/bg_img/광명시1_1000m.png")[:, :, :3]

# 색조-채도-명조로 변환
hsvImage = cv2.cvtColor(bg_img, cv2.COLOR_RGB2HSV)

# 채도 낮추기
hsvImage[:, :, 1] *= 0.1
# 명도 높이기
hsvImage[:, :, 2] = np.minimum(hsvImage[:, :, 2] * 1.02, 1.0)
# 역변환
bg_img = cv2.cvtColor(hsvImage, cv2.COLOR_HSV2RGB)

offset = [935750, 1933100]
scale = 9.2


@callback(
    Output("tgt-site-graph", "src"),
    Output("right-tabs", "value"),
    Input("graph-map", "clickData"),
    Input("input-time-span", "value"),
    Input("input-radius", "value"),
    Input("input-gross-floor", "value"),
    Input("site-graph-apply", "n_clicks"),
)
def update_site_map(clickData, timespan, radius, gross_floor, *args):
    if (
        clickData
        and "points" in clickData
        and (len(clickData["points"]) > 0)
        and clickData["points"][0]["curveNumber"] == 0
    ):
        timespan = int(timespan)
        radius = int(radius)
        map_pad = radius + figpad
        site_i = clickData["points"][0]["customdata"][0]
        # 그림 그리고자 하는 대상 공사지
        tgt_site = site_df.iloc[site_i]

        site_year = tgt_site["year"]
        site_month = tgt_site["month"]
        site_x = tgt_site["utmk_x"]
        site_y = tgt_site["utmk_y"]

        # 공사 총 개월수 계산 (준공일이 없다면 1개월로 취급)
        site_duration = get_site_duration(tgt_site, site_year, site_month)

        # 공사 시점으로부터 이웃한 n개월 기간을 구한다.
        start_year, start_month, end_year, end_month = get_acci_range(
            site_year, site_month, timespan_month=timespan
        )

        # 이전 n개월에 일어난 모든 사고를 가져온다.
        former_acci_df = acci_df[
            mask_start_date(
                acci_df, start_year, start_month, site_year, site_month, exclusive=True
            )
        ]
        # 그 중 거리상 가까운 모든 사고를 가져온다.
        former_acci_nearby_df = former_acci_df[
            mask_distance(former_acci_df, site_x, site_y, d=radius)
        ]

        # (공사년월을 포함하여) 이후 n개월이 일어난 모든 사고를 가져온다.
        latter_acci_df = acci_df[
            mask_start_date(acci_df, site_year, site_month, end_year, end_month, exclusive=True)
        ]
        # 그 중 거리상 가까운 모든 사고를 가져온다.
        latter_acci_nearby_df = latter_acci_df[
            mask_distance(latter_acci_df, site_x, site_y, d=radius)
        ]

        # 시간에 따른 범위 내 사고 카운트 (accident count timeseries)
        acci_cnt_ts = []
        current_y, current_m = start_year, start_month
        tgt_df = former_acci_nearby_df
        for i in range(timespan * 2):
            # former 범위 벗어나면 latter로 바꾸어주기.
            # if i == timespan:
            if (current_y, current_m) == (site_year, site_month):
                tgt_df = latter_acci_nearby_df
            acci_cnt_ts.append(
                tgt_df[(tgt_df["year"] == current_y) & (tgt_df["month"] == current_m)].shape[0]
            )
            current_y, current_m = current_y + (current_m // 12), (current_m % 12) + 1

        # 지도 그림의 x, y 범위
        xlim = [site_x - map_pad, site_x + map_pad]
        ylim = [site_y - map_pad, site_y + map_pad]

        fig = plt.figure(num=1, figsize=(13, 17), clear=True)
        gs = fig.add_gridspec(2, 1, height_ratios=(10, 48))
        axs = gs.subplots()

        # 상단 그림:
        # - 시간에 따른 이웃한 사고 카운트
        # - 시간에 따른 이웃한 공사의 개수
        # - 타겟 공사의 범위
        plt.sca(axs[0])
        plt.grid(True, axis="y", zorder=2)

        ###########################################3
        # 시간에 따른 이웃한 사고 카운트 시각화
        plt.bar(
            np.arange(-timespan, timespan) + 0.5,
            acci_cnt_ts,
            width=0.85,
            zorder=20,
            color="C0",
        )
        # 정확한 수 어노테이션
        for h_i in range(len(acci_cnt_ts)):
            plt.annotate(
                f"{acci_cnt_ts[h_i]}",
                [-timespan + h_i + 0.5, acci_cnt_ts[h_i]],
                ha="center",
                va="bottom",
                zorder=100,
                fontproperties=font_hangeul,
                fontsize=11.5 * font_scale,
            )

        # 이전,이후 각각의 범위 내 사고의 합
        before_after = [sum(acci_cnt_ts[:timespan]), sum(acci_cnt_ts[timespan:])]
        # 위의 것 시각화
        plt.bar(
            [-timespan / 2, timespan / 2],
            before_after,
            width=timespan / 2,
            zorder=10,
            color="skyblue",
        )
        # 정확한 수 어노테이션
        plt.annotate(
            f"{before_after[0]}",
            [-timespan / 2, before_after[0]],
            ha="center",
            va="top",
            zorder=100,
            fontproperties=font_hangeul,
            fontsize=11.5 * font_scale,
        )
        plt.annotate(
            f"{before_after[1]}",
            [timespan / 2, before_after[1]],
            ha="center",
            va="top",
            zorder=100,
            fontproperties=font_hangeul,
            fontsize=11.5 * font_scale,
        )

        # x=0 붉은 선 = 목표 공사의 시간 기준점
        plt.axvline(x=0, c="red", zorder=3)
        # 목표 공사의 범위 표시
        # plt.axvspan(xmin=0, xmax=timespan*2+1, fc=(1.0, 0.0, 0.0, 0.25), zorder=1)
        plt.axvspan(
            xmin=0,
            xmax=site_duration,
            fc=(1.0, 0.3, 0.3),
            hatch="////",
            edgecolor=(0.8, 0.2, 0.2),
            alpha=0.2,
            zorder=1,
        )

        #####################################
        # 이웃한 공사 정보 시각화
        # 이웃한 공사 필터링 - 연면적, 거리
        filtered_site_df = site_df[site_df["연면적(㎡)"] >= float(gross_floor)]
        other_site_df = filtered_site_df[
            mask_distance(filtered_site_df, site_x, site_y, d=radius)
        ]
        # 그 중 공사 시작 시점이 그리는 범위의 끝보다 이전 (즉, site_date + n개월 시점보다 이전) 공사들만 필터링
        other_site_df = other_site_df[
            mask_start_date(other_site_df, 2006, 1, end_year, end_month, exclusive=True)
        ]

        # 시간에 따른 (진행중인) 공사 개수 (site count timeseries)
        site_cnt_ts = [0] * (timespan * 2)

        other_vspans = []
        drawing_tgt_df = []
        for other_i in range(other_site_df.shape[0]):
            # 다른 공사지
            other_site = other_site_df.iloc[other_i]
            other_year, other_month = other_site["year"], other_site["month"]
            # 다른 공사지의 공사 기간
            other_duration = get_site_duration(other_site, other_year, other_month)

            # 다른 공사와 타겟 공사의 시간 차이. (ex. -6 = 6개월 이전)
            other_x = (other_year - site_year) * 12 + (other_month - site_month)

            # 다른 공사의 끝이(x+duration)이 시각화 범위의 시작 (x=-timespan)을 넘으면
            # 시각화 대상이므로 한다. (공사 시작은 이미 이전으로 보장되기 때문에 반드시 시각화 대상에 포함됨.)
            if other_x + other_duration >= -timespan:
                other_vspans.append((other_x, other_x + other_duration))
                drawing_tgt_df.append(other_site)
                for cnter_idx in range(
                    max(-timespan, other_x), min(timespan, other_x + other_duration)
                ):
                    site_cnt_ts[timespan + cnter_idx] += 1
        drawing_tgt_df = pd.DataFrame(drawing_tgt_df)
        # for x in other_vspans:
        #    plt.axvspan(xmin=x[0], xmax=x[1], fc=(0.35, 1.0, 0.35, 0.15/max(1, *site_cnt_ts)), zorder=0)

        ###########################################
        # 공사 개수 plotting
        ax2 = plt.gca().twinx()
        ax2.plot(
            np.arange(-timespan, timespan) + 0.5,
            site_cnt_ts,
            color=(0.17, 0.62, 0.17),
            marker="^",
            zorder=30,
        )
        # ax2.set_ylabel("# of Const. sites nearby")
        ax2.set_ylabel(
            "범위 내 공사장의 수", fontproperties=font_hangeul, fontsize=14 * font_scale
        )

        _, ymax = ax2.get_ylim()
        ax2.set_ylim(bottom=0.0, top=ymax * 1.15)
        # 시간에 따른 이웃한 공사 정확한 숫자 어노테이션
        for cnter_idx in range(timespan * 2):
            plt.text(
                -timespan + cnter_idx + 0.5,
                ymax * 1.03,
                f"{site_cnt_ts[cnter_idx]}",
                ha="center",
                va="center",
                c=(0.17, 0.62, 0.17),
                zorder=100,
                fontdict={"size": 11.5 * font_scale},
            )
        # 이웃한 공사 숫자 label 붙이기
        # plt.text(-timespan, max([sum(acci_cnt_ts[:timespan]), sum(acci_cnt_ts[timespan:])])*1.02, f"# of Const. sites",
        #         ha="right", va="center", c=(0.17, 0.62, 0.17), zorder=1000, bbox={"facecolor":(1.0, 0.0, 0.0, 1.0), "boxstyle":"round", })

        # plt.annotate(f"# of Const. sites", [-timespan, ymax*1.03],
        #       ha="right", va="center", c=(0.17, 0.62, 0.17), zorder=100, bbox={"facecolor":(1.0, 1.0, 1.0, 0.75), "boxstyle":"round", })
        plt.annotate(
            f"공사의 수",
            [-timespan, ymax * 1.03],
            ha="right",
            va="center",
            c=(0.17, 0.62, 0.17),
            zorder=100,
            bbox={
                "facecolor": (1.0, 1.0, 1.0, 0.75),
                "boxstyle": "round",
            },
            fontproperties=font_hangeul,
            fontsize=11.5 * font_scale,
        )
        plt.sca(axs[0])

        ############################################
        # 타겟 공사 시작 시간 / 끝 시간 표기, 그외 기타 정보.
        # plt.text(0.49, 0.99, f"Const. start: {tgt_site['year']}{tgt_site['month']:02d}",
        #          ha="right", va="top", transform =plt.gca().transAxes, zorder=11)
        # plt.text(0.99, 0.99, f"Const. End: {tgt_site['사용승인일']:.0f}",
        #          ha="right", va="top", transform =plt.gca().transAxes, zorder=11)
        # plt.xticks(np.arange(-timespan, timespan+1), np.arange(-timespan,timespan+1))
        # plt.title(f"Accident count before/after construction #{site_i}\nradius: {radius}m", fontsize=20)
        # plt.ylabel("# of Accidents nearby")
        # plt.xlabel("$\Delta$months")
        plt.text(
            0.49,
            0.99,
            f"공사시작: {tgt_site['year']}{tgt_site['month']:02d}",
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
            zorder=11,
            fontproperties=font_hangeul,
            fontsize=11 * font_scale,
        )
        plt.text(
            0.99,
            0.99,
            f"공사종료:" + f"{tgt_site['사용승인일']:.0f}"[:6],
            ha="right",
            va="top",
            transform=plt.gca().transAxes,
            zorder=11,
            fontproperties=font_hangeul,
            fontsize=11 * font_scale,
        )
        plt.xticks(np.arange(-timespan, timespan + 1), np.arange(-timespan, timespan + 1))
        plt.title(
            f"공사 전/후 사고빈도 비교 - 샘플 #{site_i}\n이웃반경: {radius}m, 공사의 연면적 {gross_floor}㎡ 이상",
            fontsize=20 * font_scale,
            fontproperties=font_hangeul,
        )
        plt.ylabel("사고빈도", fontproperties=font_hangeul, fontsize=14 * font_scale)
        plt.xlabel("$\Delta$months", fontproperties=font_hangeul, fontsize=14 * font_scale)

        # x축 범위, y축 범위 여유좀 주기
        plt.xlim([-0.6 - timespan, timespan + 0.6])
        _, ymax = plt.gca().get_ylim()
        plt.ylim(top=ymax * 1.1)

        # legend 처리. (보이지 않는 vspan 활용)
        # other_vspan = plt.axvspan(xmin=0, xmax=0, fc=(0.35, 0.35, 1.0),label="Other Const. period")
        # tmp_bar = plt.bar([0], [0], width=0.0, color="C0", label="Accidents nearby")
        # tmp_sum_bar = plt.bar([0], [0], width=0.0, color="skyblue", label="Total accident count (Before/After)")
        # other_plot = plt.plot([0], [0], color=(0.17, 0.62, 0.17), marker="^", label="Const. sites nearby")[0]
        # target_vspan = plt.axvspan(xmin=0, xmax=0, fc=(1.0, 0.2, 0.2), hatch="////", edgecolor=(0.8, 0.2, 0.2), alpha=0.4, label="Target Const. period")
        # plt.legend(handles=(tmp_bar, tmp_sum_bar, other_plot, target_vspan),
        #            loc="upper right", bbox_to_anchor=(1.0, -0.1)).set_zorder(100)

        tmp_bar = plt.bar([0], [0], width=0.0, color="C0", label="근처에서 발생한 사고")
        tmp_sum_bar = plt.bar(
            [0], [0], width=0.0, color="skyblue", label="공사 전/후 총 사고빈도"
        )
        other_plot = plt.plot(
            [0], [0], color=(0.17, 0.62, 0.17), marker="^", label="이웃한 공사 개수"
        )[0]
        target_vspan = plt.axvspan(
            xmin=0,
            xmax=0,
            fc=(1.0, 0.2, 0.2),
            hatch="////",
            edgecolor=(0.8, 0.2, 0.2),
            alpha=0.4,
            label=f"샘플 공사의 기간",
        )
        plt.legend(
            handles=(tmp_bar, tmp_sum_bar, other_plot, target_vspan),
            loc="upper right",
            bbox_to_anchor=(1.0, -0.1),
            prop=font_hangeul_legend,
        ).set_zorder(100)

        ##############################################
        # 하단 그림: 타겟 공사의 주변 사고/공사
        plt.sca(axs[1])
        plt.imshow(
            bg_img,
            extent=(
                offset[0],
                offset[0] + bg_img.shape[1] * scale,
                offset[1],
                offset[1] + bg_img.shape[0] * scale,
            ),
            alpha=0.5,
        )

        former_mask = mask_end_date(
            drawing_tgt_df, start_year, start_month, site_year, site_month, exclusive=True
        )
        latter_mask = mask_start_date(
            drawing_tgt_df,
            site_year + site_month // 12,
            site_month % 12 + 1,
            end_year,
            end_month,
            exclusive=True,
        )
        coexist_mask = ~(former_mask | latter_mask)

        # 이전에 끝난 공사
        former_site_df = drawing_tgt_df[former_mask]

        # 동시간에 진행되는 공사
        coexist_site_df = drawing_tgt_df[coexist_mask]
        # 이후에 시작한 공사
        latter_site_df = drawing_tgt_df[latter_mask]
        ax = plt.gca()
        plt.scatter(
            former_acci_df["utmk_x"],
            former_acci_df["utmk_y"],
            s=20 * dot_scale,
            c=[[0.0, 0, 1.0, 0.3]],
            marker="X",
            zorder=10,
            # label = "Accident before target const.",
            label="샘플 공사 착공 이전의 사고",
        )
        plt.scatter(
            latter_acci_df["utmk_x"],
            latter_acci_df["utmk_y"],
            s=20 * dot_scale,
            c=[[1.0, 0, 0.0, 0.3]],
            marker="X",
            zorder=10,
            # label = "Accident after target const.",
            label="샘플 공사 착공 이후의 사고",
        )

        plt.scatter(
            former_site_df["utmk_x"],
            former_site_df["utmk_y"],
            s=5 * dot_scale,
            c=[[0.0, 0, 1.0, 1.0]],
            edgecolor=[[1.0, 1.0, 1.0, 0.6]],
            linewidths=2,
            zorder=10,
            # label = "Other const. before target const.",
            label="샘플 공사 착공 이전의 끝난 다른 공사",
        )

        plt.scatter(
            coexist_site_df["utmk_x"],
            coexist_site_df["utmk_y"],
            s=5 * dot_scale,
            c=[[0.85, 0.6, 0.0, 1.0]],
            edgecolor=[[1.0, 1.0, 1.0, 0.6]],
            linewidths=2,
            zorder=10,
            # label = "Other const. at the same time",
            label="샘플 공사 착공 시점에 진행된 다른 공사",
        )

        plt.scatter(
            [site_x],
            [site_y],
            s=50 * dot_scale,
            c=[[1.0, 1.0, 0, 1.0]],
            edgecolor=[[1.0, 0.0, 0.0, 1.0]],
            marker="o",
            zorder=15,
            # label = "Target const.",
            label="샘플 공사",
        )
        plt.annotate(
            f"{tgt_site['대지위치'][:18]}",
            xy=[site_x, site_y],
            xytext=[30, 40],
            textcoords="offset pixels",
            fontproperties=font_hangeul,
            ha="left",
            arrowprops=dict(
                arrowstyle="-",
                connectionstyle="arc,angleA=-90,angleB=0,armA=20,armB=0,rad=0",
                relpos=(0.25, 0.0),
            ),
            zorder=14,
        )

        plt.scatter(
            latter_site_df["utmk_x"],
            latter_site_df["utmk_y"],
            s=5 * dot_scale,
            c=[[1.0, 0, 0, 1.0]],
            edgecolor=[[1.0, 1.0, 1.0, 0.6]],
            linewidths=2,
            zorder=10,
            # label = "Other const. after target const.",
            label="샘플 공사 착공 이후 시작한 다른 공사",
        )

        for x, y in former_site_df[["utmk_x", "utmk_y"]].values:
            circ = patches.Circle(
                (x, y), radius=radius, fc=(0.0, 0.0, 0.0, 0.0), ec=(0.0, 0.0, 1.0, 0.08)
            )
            ax.add_patch(circ)
        for x, y in coexist_site_df[["utmk_x", "utmk_y"]].values:
            circ = patches.Circle(
                (x, y),
                radius=radius,
                fc=(0.85, 0.0, 0.6, 0.0),
                ec=(1.0, 0.0, 0.0, 0.12),
            )
            ax.add_patch(circ)
        for x, y in latter_site_df[["utmk_x", "utmk_y"]].values:
            circ = patches.Circle(
                (x, y),
                radius=radius,
                fc=(0.0, 0.0, 0.0, 0.0),
                ec=(0.85, 0.6, 0.0, 0.08),
            )
            ax.add_patch(circ)

        circ = patches.Circle(
            (site_x, site_y),
            radius=radius,
            fc=(0.0, 0.0, 0.0, 0.00),
            ec=(1.0, 0.0, 0.4, 0.8),
            lw=3,
        )
        ax.add_patch(circ)

        plt.legend(loc=4, fontsize=14 * font_scale, prop=font_hangeul_legend2).set_zorder(100)
        plt.xlim(xlim)
        plt.ylim(ylim)
        plt.xlabel("UTM-K X", fontsize=12 * font_scale)
        plt.ylabel("UTM-K Y", fontsize=12 * font_scale)
        plt.gca().set_aspect("equal")
        plt.title(
            f"{start_year}-{start_month:02d}~{end_year}-{end_month:02d}\n"
            f"{tgt_site['대지위치'][:33]}",
            fontproperties=font_hangeul,
            fontsize=20 * font_scale,
        )

        plt.tight_layout()

        # plot 띄우기
        buf = BytesIO()
        plt.savefig(buf, format="png")

        fig_data = base64.b64encode(buf.getbuffer()).decode("ascii")
        fig_matplotlib = f"data:image/png;base64,{fig_data}"

        return fig_matplotlib, "site-graph-tab"
    else:
        raise PreventUpdate


if __name__ == "__main__":
    app.run(debug=True)
