import json
import datetime

from dash import (
    Dash,
    html,
    dcc,
    Input,
    Output,
    State,
    callback,
    dash_table,
    Patch,
    ctx,
    no_update,
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


gcs_to_utmk = Transformer.from_crs("EPSG:4326", "EPSG:5178", always_xy=True)

site_df = pd.read_csv("./outputs/공사_통합처리.csv")
acci_df = pd.read_csv("./data/사고지점_통합_2007_2023.csv")

acci_df["color"] = [0] * acci_df.shape[0]

app = Dash(__name__)

# max_lon = 126.9023
# min_lon = 126.8285
# max_lat = acci_df["latitude"].max()
# min_lat = acci_df["latitude"].min()

# tmp_data = [
#     [x, y, 1]
#     for x in np.arange(min_lon, max_lon, 0.0002)
#     for y in np.arange(min_lat, max_lat, 0.0002)
# ]

tmp_data = [[126.865638714, 37.4594013977, 1]]

tmp_df = pd.DataFrame(tmp_data, columns=["longitude", "latitude", "color"])
tmp_utmk_x, tmp_utmk_y = gcs_to_utmk.transform(tmp_df["longitude"], tmp_df["latitude"])
tmp_df["utmk_x"] = tmp_utmk_x
tmp_df["utmk_y"] = tmp_utmk_y

fig = px.scatter_map(
    acci_df,
    lat="latitude",
    lon="longitude",
    color="color",
    color_continuous_scale="jet",
    hover_data=["utmk_x", "utmk_y"],
    zoom=12,
    height=1000,
)

tmp_fig = px.scatter_map(
    tmp_df,
    lat="latitude",
    lon="longitude",
    color="color",
    hover_data=["utmk_x", "utmk_y"],
    zoom=12,
    height=1000,
)

fig.add_traces(tmp_fig.data)


selected_fig = px.scatter_map(
    {"latitude": [], "longitude": [], "color": [], "utmk_x": [], "utmk_y": []},
    lat="latitude",
    lon="longitude",
    color="color",
    hover_data=["utmk_x", "utmk_y"],
    zoom=12,
    height=1000,
)

fig.add_traces(selected_fig.data)

app.layout = html.Div(
    [
        html.Button(
            children="되돌리기",
            id="button-undo",
            style={"width": "150px", "height": "40px", "margin": "10px"},
        ),
        html.Button(
            children="내보내기",
            id="button-export",
            style={"width": "150px", "height": "40px", "margin": "10px"},
        ),
        # dcc.Graph(
        #     figure=px.scatter_map(
        #         site_df,
        #         lat="위도",
        #         lon="경도",
        #         zoom=12,
        #         height=1000,
        #     ),
        #     id="graph-map",
        # )
        dcc.Graph(
            figure=fig,
            id="graph-map",
        ),
    ]
)


# 지점 선택 후 저장
selected_points_df = pd.DataFrame(
    [], columns=["longitude", "latitude", "color", "utmk_x", "utmk_y"]
)

print(fig.data[2])


@callback(
    Output("graph-map", "figure", allow_duplicate=True),
    Input("graph-map", "relayoutData"),
    prevent_initial_call=True,
)
def display_relayout_data(relayoutData):
    print(relayoutData)
    if relayoutData and ("map.center" in relayoutData):
        tmp_data = [[relayoutData["map.center"]["lon"], relayoutData["map.center"]["lat"], 1]]
        tmp_data = [
            [tmp_data[0][0] + dx, tmp_data[0][1] + dy, 1]
            for dx in [-0.0002, 0.0, 0.0002]
            for dy in [-0.0002, 0.0, 0.0002]
        ]
        tmp_df = pd.DataFrame(tmp_data, columns=["longitude", "latitude", "color"])
        tmp_utmk_x, tmp_utmk_y = gcs_to_utmk.transform(tmp_df["longitude"], tmp_df["latitude"])
        tmp_df["utmk_x"] = tmp_utmk_x
        tmp_df["utmk_y"] = tmp_utmk_y
        fig_patch = Patch()
        fig_patch["data"][1]["lat"] = tmp_df["latitude"].to_numpy()
        fig_patch["data"][1]["lon"] = tmp_df["longitude"].to_numpy()
        fig_patch["data"][1]["customdata"] = tmp_df.loc[:, ["utmk_x", "utmk_y"]].to_numpy()
        fig_patch["data"][1]["marker"]["color"] = tmp_df["color"].to_numpy(dtype=np.int64)
        return fig_patch
    return no_update


def updated_selected_scatter_patch():
    global selected_points_df
    fig_patch = Patch()
    fig_patch["data"][2]["lat"] = selected_points_df["latitude"].to_numpy()
    fig_patch["data"][2]["lon"] = selected_points_df["longitude"].to_numpy()
    fig_patch["data"][2]["customdata"] = selected_points_df.loc[
        :, ["utmk_x", "utmk_y"]
    ].to_numpy()
    fig_patch["data"][2]["marker"]["color"] = selected_points_df["color"].to_numpy(
        dtype=np.float32
    )
    return fig_patch


@callback(
    Output("graph-map", "figure", allow_duplicate=True),
    Input("graph-map", "clickData"),
    prevent_initial_call=True,
)
def print_clickData(clickData):
    global selected_points_df
    if clickData:
        # print(clickData)
        data_point = [
            clickData["points"][0]["lon"],
            clickData["points"][0]["lat"],
            0.8,
            *clickData["points"][0]["customdata"],
        ]
        print(f"지점 {len(selected_points_df)+1}")
        print()
        print("경도:", clickData["points"][0]["lon"])
        print("위도:", clickData["points"][0]["lat"])
        print("utmk_x:", clickData["points"][0]["customdata"][0])
        print("utmk_y:", clickData["points"][0]["customdata"][1])
        print()
        selected_points_df.loc[selected_points_df.shape[0]] = data_point
        fig_patch = updated_selected_scatter_patch()
        return fig_patch
    return no_update


@callback(
    Input("button-export", "n_clicks"),
    prevent_inital_call=True,
)
def export_risky_spot(*args):
    global selected_points_df
    selected_points_df.drop(columns=["color"]).to_csv("안전도지점.csv", index=False)


@callback(
    Output("graph-map", "figure", allow_duplicate=True),
    Input("button-undo", "n_clicks"),
    prevent_initial_call=True,
)
def export_risky_spot(*args):
    global selected_points_df
    if selected_points_df.shape[0] > 0:
        selected_points_df.drop(index=selected_points_df.index[-1], inplace=True)
        fig_patch = updated_selected_scatter_patch()
        return fig_patch
    return no_update


if __name__ == "__main__":
    app.run(debug=True)
