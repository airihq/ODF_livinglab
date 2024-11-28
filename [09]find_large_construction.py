import datetime
import os

from dash import Dash, html, dcc, Input, Output, State, callback, dash_table
from dash.exceptions import PreventUpdate
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


os.makedirs("./outputs", exist_ok=True)


acci_df = pd.read_csv("./data/사고지점_통합_2007_2023.csv")

site_df = pd.read_csv("./data/지오코딩_결과.csv")

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

site_df["year"], site_df["month"], site_df["day"] = (
    site_date.apply(lambda x: int(x[:4])),
    site_date.apply(lambda x: int(x[4:6])),
    site_date.apply(lambda x: int(x[6:8])),
)

site_df["timestamp"] = site_date.apply(
    lambda x: datetime.datetime(
        int(x[:4]),
        int(x[4:6]),
        int(x[6:8]),
        tzinfo=datetime.timezone(offset=datetime.timedelta(hours=-9)),
    ).timestamp()
)

time_ticks = [f"{year}-{month:02d}" for year in range(2007, 2023) for month in range(1, 13)]


###############
# 레이블 생성
site_df.insert(site_df.shape[1], "labels", site_df["주용도"])
# 예외처리
site_df.loc[pd.isna(site_df["labels"]), "labels"] = ""
# 기타용도에 아파트라면 아파트 레이블 추가
for i in site_df.index:
    if (not pd.isna(site_df.loc[i, "기타용도"])) and ("아파트" in site_df.loc[i, "기타용도"]):
        site_df.loc[i, "labels"] = site_df.loc[i, "labels"] + ",아파트"

site_df = site_df.sort_values("timestamp", axis=0)

## 나중에 datapoints로 저장하여 접근하기 쉽도록 인덱스 저장
# site_df.insert(0, "index", list(range(len(site_df))))


def clustring(datapoints, distance=100):
    n = len(datapoints)
    if n == 0:
        return []
    if n == 1:
        return [datapoints]

    cluster_id = list(range(n))
    for i in range(n):
        src = datapoints[i]
        for j in range(i + 1, n):
            tgt = datapoints[j]
            if (src["utmk_x"] - tgt["utmk_x"]) ** 2 + (
                src["utmk_y"] - tgt["utmk_y"]
            ) ** 2 <= distance**2:
                cluster_id[j] = cluster_id[i]
    remain_ids = np.unique(cluster_id)
    res_clusters = {k: [] for k in remain_ids}
    for i, id in enumerate(cluster_id):
        res_clusters[id].append(datapoints[i])
    res_clusters = [res_clusters[k] for k in res_clusters]
    return res_clusters


def get_all_candi_clusters(site_df, distance, cluster_size):
    candi_clusters = []

    # 동시간의 데이터포인트를 모으로 clustring 함수로 보냄
    datapoints = []
    current_time = None
    for i in range(site_df.shape[0]):
        sample = site_df.iloc[i]
        if not current_time or current_time != sample["timestamp"]:
            # 클러스터링
            candi_clusters += clustring(datapoints, distance)
            current_time = sample["timestamp"]
            datapoints = [sample]
        else:
            datapoints.append(sample)
    if datapoints:
        candi_clusters += clustring(datapoints)
    candi_clusters = [c for c in candi_clusters if len(c) >= cluster_size]
    return candi_clusters


app = Dash(__name__)


input_panels = html.Div(
    [
        html.Div(
            children="대규모 클러스터 기준 (공사개수)",
            style={
                "width": "350px",
                "height": "fit-content",
                "text-align": "right",
                "padding": "10px",
                # "position": "relative",
                # "top": "50%",
                # "transform": "translate(0, -50%)",
            },
        ),
        dcc.Input(
            value="3",
            type="number",
            placeholder="공사 개수",
            id="input-site-num",
            debounce=True,
            style={
                "width": "120px",
            },
        ),
        html.Div(
            children="인접한 공사의 거리 (m)",
            style={
                "width": "350px",
                "height": "fit-content",
                "text-align": "right",
                "padding": "10px",
                # "position": "relative",
                # "top": "50%",
                # "transform": "translate(0, -50%)",
            },
        ),
        dcc.Input(
            value="50",
            type="number",
            placeholder="단위: m",
            id="input-dist-nearby",
            debounce=True,
            style={
                "width": "120px",
                "height": "fit-content",
            },
        ),
        html.Div(
            children="아파트는 반드시 대규모 공사인가요?",
            style={
                "width": "350px",
                "height": "fit-content",
                "text-align": "right",
                "padding": "10px",
                # "position": "relative",
                # "top": "50%",
                # "transform": "translate(0, -50%)",
            },
        ),
        dcc.Checklist(["Yes"], value=["Yes"], id="is-apt-large"),
        html.Div(
            children="통합을 안해도 대규모 레이블을 부여",
            style={
                "width": "350px",
                "text-align": "right",
                "padding": "10px",
                # "position": "relative",
                # "top": "50%",
                # "transform": "translate(0, -50%)",
            },
        ),
        dcc.Checklist(["Yes"], value=["Yes"], id="keep-large-label"),
        html.Div(
            [
                html.Button(
                    children="검색 개시",
                    id="button-search",
                    style={"width": "150px", "height": "40px", "margin": "10px"},
                ),
            ],
            style={
                "grid-column": "1 / 3",
                "text-align": "center",
                "align-content": "center",
            },
        ),
        html.Div(children="", id="cluster-counter"),
        html.Div(
            [
                html.Button(
                    children="통합",
                    id="button-combine",
                    disabled=True,
                    style={"width": "150px", "height": "40px", "margin": "10px"},
                ),
                html.Button(
                    children="넘어가기",
                    id="button-skip",
                    disabled=True,
                    style={"width": "150px", "height": "40px", "margin": "10px"},
                ),
            ],
            style={
                "grid-column": "1 / 3",
                "text-align": "left",
                "align-content": "center",
            },
        ),
    ],
    style={
        "display": "grid",
        "grid-template-columns": "max-content auto",
        "grid-template-rows": "repeat(7, 1fr)",
        "align-items": "center",
    },
)
map_panels = dcc.Graph(id="graph-cluster-map")

app.layout = html.Div(
    [
        html.Div(
            [input_panels, map_panels],
            style={
                "display": "grid",
                "grid-template-columns": "60% minmax(40%, 500px)",
                # "align-items": "start",
                # "justify-items": "start",
            },
        ),
        html.Div(
            [
                dash_table.DataTable(
                    columns=[
                        {"id": i, "name": i}
                        for i in site_df.columns
                        if i not in ["utmk_x", "utmk_y", "year", "month", "day", "timestamp"]
                    ],
                    page_size=10,
                    style_data={"whiteSpace": "normal", "height": "auto"},
                    style_table={"overflowX": "auto"},
                    id="candi-table",
                )
            ],
            style={"width": "fit-content"},
        ),
    ],
    style={
        "display": "grid",
        "grid-template-rows": "1fr 1fr",
        # "align-items": "start",
        # "justify-items": "start",
    },
)


gb_candi_cluster = None
gb_is_apt_large = True
gb_keep_large_label = False
gb_cluster_idx = -1


@callback(
    Output("input-site-num", "disabled"),
    Output("input-dist-nearby", "disabled"),
    Output("is-apt-large", "options"),
    Output("keep-large-label", "options"),
    Output("button-search", "disabled"),
    Output("button-combine", "disabled"),
    Output("button-skip", "disabled"),
    Output("cluster-counter", "children", allow_duplicate=True),
    State("input-site-num", "value"),
    State("input-dist-nearby", "value"),
    State("is-apt-large", "value"),
    State("keep-large-label", "value"),
    Input("button-search", "n_clicks"),
    prevent_initial_call=True,
)
def start_clustering(cluster_size, distance, is_apt_large, keep_large_label, *args):
    global gb_candi_cluster, gb_is_apt_large, gb_keep_large_label, gb_cluster_idx
    cluster_size = int(cluster_size)
    distance = float(distance)
    gb_is_apt_large = "Yes" in is_apt_large
    gb_keep_large_label = "Yes" in keep_large_label

    # 아파트레이블에 대규모 추가
    if gb_is_apt_large:
        for i in site_df.index:
            if (not pd.isna(site_df.loc[i, "기타용도"])) and (
                "아파트" in site_df.loc[i, "기타용도"]
            ):
                site_df.loc[i, "labels"] = site_df.loc[i, "labels"] + ",대규모"

    gb_candi_cluster = get_all_candi_clusters(
        site_df, distance=distance, cluster_size=cluster_size
    )
    gb_cluster_idx = 0
    return (
        True,
        True,
        [{"label": "Yes", "value": "Yes", "disabled": True}],
        [{"label": "Yes", "value": "Yes", "disabled": True}],
        True,
        False,
        False,
        f"{gb_cluster_idx}/{len(gb_candi_cluster)}",
    )


@callback(
    Output("graph-cluster-map", "figure"),
    Output("candi-table", "data"),
    Input("cluster-counter", "children"),
    prevent_initial_call=True,
)
def update_graph_and_table(*args):
    global gb_cluster_idx, gb_candi_cluster
    if gb_cluster_idx < 0:
        raise PreventUpdate
    if gb_cluster_idx >= len(gb_candi_cluster):
        site_df.drop(
            columns=["utmk_x", "utmk_y", "year", "month", "day", "timestamp"], inplace=True
        )
        site_df.to_csv("./outputs/공사_통합처리.csv", index=False)
        raise PreventUpdate
    site_cluster = gb_candi_cluster[gb_cluster_idx]
    cluster_df = pd.DataFrame(site_cluster)
    cluster_df["color"] = np.array(["통합대상" for _ in range(len(cluster_df))])
    cluster_df["size"] = np.array([8 for _ in range(len(cluster_df))])

    fig = px.scatter_map(
        cluster_df,
        lat="위도",
        lon="경도",
        size="size",
        color="color",
        color_discrete_map={"통합대상": "#FF3131"},
        zoom=14,
        opacity=0.16,
    )
    return fig, cluster_df.to_dict("records")


@callback(
    Output("cluster-counter", "children", allow_duplicate=True),
    Input("button-combine", "n_clicks"),
    prevent_initial_call=True,
    running=[
        (Output("button-combine", "disabled"), True, False),
        (Output("button-skip", "disabled"), True, False),
    ],
)
def combine_cluster(*args):
    global gb_cluster_idx, gb_candi_cluster, site_df
    if gb_cluster_idx >= len(gb_candi_cluster):
        raise PreventUpdate
    site_cluster = gb_candi_cluster[gb_cluster_idx]
    cluster_df = pd.DataFrame(site_cluster)
    # label 합치기
    labels_set = set()
    for tmp_labels in cluster_df["labels"]:
        labels_set.update(set(tmp_labels.split(",")))
    if len(site_cluster) >= 3:
        labels_set.update({"대규모"})
    print(labels_set)

    ### 클러스터에서 가장 앞 레코드만 남기고 그거 수정한 뒤 나머지 다 지우기
    # 평균 좌표
    mean_lat = cluster_df["위도"].mean()
    mean_lon = cluster_df["경도"].mean()
    mean_x, mean_y = gcs_to_utmk.transform(mean_lon, mean_lat)

    # 평균 좌표에서 가장 가까운 주소를 사용하기
    cluster_df["dist"] = (
        cluster_df["utmk_x"].apply(lambda x: (x - mean_x) ** 2).to_numpy()
        + cluster_df["utmk_y"].apply(lambda y: (y - mean_y) ** 2).to_numpy()
    )
    repr_address = cluster_df[cluster_df["dist"] == cluster_df["dist"].min()].iloc[0]["대지위치"]
    # 착공예정일과 실착공일은 가장 이른 시점으로
    expected_date = cluster_df["착공예정일"].min()
    started_date = cluster_df["실착공일"].min()
    # 사용승인일은 가장 마지막에
    end_date = cluster_df["사용승인일"].max()
    # 나머지는... 모르겠다 그냥 연면적이 가장 넓은 규모의 공사 정보를 전부 가져온다.
    largest_sample = cluster_df[cluster_df["연면적(㎡)"] == cluster_df["연면적(㎡)"].max()].iloc[
        0
    ]

    tgt_idx = cluster_df.index[0]
    site_df.loc[tgt_idx, "위도"] = mean_lat
    site_df.loc[tgt_idx, "경도"] = mean_lon
    site_df.loc[tgt_idx, "대지위치"] = repr_address
    site_df.loc[tgt_idx, "착공예정일"] = expected_date
    site_df.loc[tgt_idx, "실착공일"] = started_date
    site_df.loc[tgt_idx, "사용승인일"] = end_date
    site_df.loc[tgt_idx, "labels"] = ",".join(labels_set)

    other_columns = [
        "대지면적(㎡)",
        "건축면적(㎡)",
        "건폐율(%)",
        "연면적(㎡)",
        "용적률산정용면적(㎡)",
        "용적률(%)",
        "주용도",
        "기타용도",
    ]

    site_df.loc[tgt_idx, other_columns] = largest_sample[other_columns]

    site_df = site_df.drop(index=cluster_df.index[1:])

    # 다음 클러스터로 넘어가기
    gb_cluster_idx += 1

    return (f"{gb_cluster_idx}/{len(gb_candi_cluster)}",)


@callback(
    Output("cluster-counter", "children", allow_duplicate=True),
    Input("button-skip", "n_clicks"),
    prevent_initial_call=True,
    running=[
        (Output("button-combine", "disabled"), True, False),
        (Output("button-skip", "disabled"), True, False),
    ],
)
def skip_cluster(*args):
    global gb_cluster_idx, gb_keep_large_label, gb_candi_cluster, site_df
    if gb_cluster_idx >= len(gb_candi_cluster):
        raise PreventUpdate
    site_cluster = gb_candi_cluster[gb_cluster_idx]
    cluster_df = pd.DataFrame(site_cluster)
    if gb_keep_large_label:
        for i in cluster_df.index:
            site_df.loc[i, "labels"] = site_df.loc[i, "labels"] + ",대규모"
    # 다음 클러스터로 넘어가기
    gb_cluster_idx += 1

    return (f"{gb_cluster_idx}/{len(gb_candi_cluster)}",)


# @callback(Output("graph-map", "figure"), Input("time-range", "value"))
# def update_map(time_range):
#     fig = draw_mapfig(time_ticks[time_range[0]], time_ticks[time_range[1]], initial=False)
#     fig.update_layout(uirevision=1)
#     fig.update_xaxes(autorange=False)
#     fig.update_yaxes(autorange=False)
#     return fig.to_dict()


if __name__ == "__main__":
    app.run(debug=True)
