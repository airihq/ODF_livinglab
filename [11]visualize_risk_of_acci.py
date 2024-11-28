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
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px
import pandas as pd
import numpy as np


risk_df = pd.read_csv("./outputs/위험도_결과.csv")

time_ticks = [f"{year}-{month:02d}" for year in range(2012, 2024) for month in range(1, 13)]

init_year = 2023
init_month = 12
dff = risk_df.loc[(risk_df["year"] == init_year) & (risk_df["month"] == init_month)].copy()
dff.loc[:, "size"] = [7 for _ in range(dff.shape[0])]
init_map_fig = px.scatter_map(
    dff,
    lat="latitude",
    lon="longitude",
    size="size",
    color="predicted",
    custom_data=["spot_idx"],
    labels={"latitude": "위도", "longitude": "경도", "predicted": "예측 위험도"},
    hover_data={"latitude": ":.6", "longitude": ":.6", "size": False, "predicted": ":.3f"},
    color_continuous_scale="jet",
    range_color=[0, 6],
    zoom=12,
)

gb_spot_idx = 0
tmp = risk_df[risk_df["spot_idx"] == gb_spot_idx].iloc[0]
analysis_dur = 12

init_map_fig.add_trace(
    go.Scattermap(
        mode="markers",
        lon=[tmp["longitude"]],
        lat=[tmp["latitude"]],
        marker={"size": 10, "symbol": ["star"]},
        showlegend=False,
        hoverinfo="skip",
    )
)

init_map_fig.update_layout(margin=dict(l=20, r=20, b=20, t=20))

app = Dash(__name__, assets_folder="./assets_11")

app.layout = html.Div(
    [
        html.H1("광명시 위험도 예측 분석", style={"text-align": "center"}),
        html.Div(
            [
                # left
                html.Div(
                    [
                        # time dropdowns
                        html.Div(
                            [
                                html.Div(
                                    children="연도",
                                    style={
                                        "text-align": "right",
                                        "position": "relative",
                                        "top": "50%",
                                        "transform": "translate(0, -50%)",
                                        "padding": "10px",
                                    },
                                ),
                                dcc.Dropdown(
                                    [i for i in range(2012, 2024)],
                                    value=2023,
                                    id="year-dropdown",
                                    clearable=False,
                                ),
                                html.Div(
                                    children="월",
                                    style={
                                        "text-align": "right",
                                        "position": "relative",
                                        "top": "50%",
                                        "transform": "translate(0, -50%)",
                                        "padding": "10px",
                                    },
                                ),
                                dcc.Dropdown(
                                    [i for i in range(1, 13)],
                                    value=12,
                                    id="month-dropdown",
                                    clearable=False,
                                ),
                            ],
                            style={
                                "display": "grid",
                                "grid-template-columns": "90px 150px 90px 150px",
                                "justify-content": "center",
                            },
                        ),
                        # time slide
                        html.Div(
                            [
                                dcc.Slider(
                                    min=0,
                                    max=len(time_ticks) - 1,
                                    value=len(time_ticks) - 1,
                                    step=1,
                                    marks={
                                        i: time_ticks[i][:4]
                                        for i in range(0, len(time_ticks), 12)
                                    },
                                    id="time-slider",
                                ),
                            ],
                            style={"width": "700px", "margin": "auto"},
                        ),
                        # map
                        dcc.Graph(figure=init_map_fig, id="risk-map", style={"height": "950px"}),
                    ],
                    style={
                        "display": "grid",
                        "grid-template-rows": "max(5%, 50px) max(5%, 50px) auto",
                        "text-align": "center",
                    },
                ),
                # right
                html.Div(
                    [
                        # 통계 그래프
                        html.H2("예측 위험도 / 사고 빈도 추세"),
                        dcc.Graph(id="risk-analysis-graph"),
                        html.H2("위험도 예측에 사용된 특징들 분석"),
                        dcc.Graph(id="features-analysis-graph"),
                        # html.Img(src=init_analysis_img, id="analysis-img", height="850px"),
                    ],
                    style={
                        "display": "grid",
                        "grid-template-rows": "10px min(200px, 1fr) 10px min(200px 1fr)",
                    },
                ),
            ],
            style={
                "display": "grid",
                "grid-template-columns": "max(200px, 70%) auto",
            },
        ),
    ]
)


@callback(
    Output("year-dropdown", "value"),
    Output("month-dropdown", "value"),
    Output("time-slider", "value"),
    Input("year-dropdown", "value"),
    Input("month-dropdown", "value"),
    Input("time-slider", "value"),
)
def validate_range(year, month, time_slider):
    # print(ctx.triggered)
    triggered_id = ctx.triggered[0]["prop_id"].split(".")[0]
    if "slider" in triggered_id:
        time_str = time_ticks[time_slider]
        year, month = (int(c) for c in time_str.split("-"))
        return year, month, no_update
    elif "dropdown" in triggered_id:
        return no_update, no_update, time_ticks.index(f"{year}-{month:02d}")
    raise PreventUpdate


@callback(
    Output("risk-map", "figure", allow_duplicate=True),
    Input("year-dropdown", "value"),
    Input("month-dropdown", "value"),
    prevent_initial_call=True,
)
def update_map(year, month):
    global dff
    year = int(year)
    month = int(month)
    dff = risk_df.loc[(risk_df["year"] == year) & (risk_df["month"] == month)]
    # dff.loc[:, "size"] = [8 for _ in range(dff.shape[0])]
    patch = Patch()
    patch["data"][0]["marker"]["color"] = dff["predicted"]
    return patch


@callback(
    # Output("analysis-img", "src"),
    Output("risk-analysis-graph", "figure"),
    Output("features-analysis-graph", "figure"),
    Output("risk-map", "figure"),
    Input("risk-map", "clickData"),
    Input("time-slider", "value"),
    # prevent_initial_call=True,
)
def click_callback(clickData, time_slider):
    global gb_spot_idx
    if (
        clickData
        and "points" in clickData
        and (len(clickData["points"]) > 0)
        and clickData["points"][0]["curveNumber"] == 0
    ):
        gb_spot_idx = clickData["points"][0]["customdata"][0]
    dff = risk_df[risk_df["spot_idx"] == gb_spot_idx].copy()
    time_x = [f"{year}-{month:02d}" for year, month in dff.loc[:, ["year", "month"]].to_numpy()]
    predicted_y = dff["predicted"].to_numpy()
    real_y = dff["acci_rate"].to_numpy()

    feat_names = [
        "이전 3달 평균 250m 사고빈도",
        "전년도 같은 달 250m 사고빈도",
        "3년 평균 250m 사고빈도",
        "1500m 내 공사 수",
        "2000m 내 대규모 공사",
    ]

    feat1 = dff[feat_names[0]].to_numpy()
    feat2 = dff[feat_names[1]].to_numpy()
    feat3 = dff[feat_names[2]].to_numpy()
    feat4 = dff[feat_names[3]].to_numpy()
    feat5 = dff[feat_names[4]].to_numpy()

    # 혹시 불일치할 경우 대비
    last_idx = time_x.index(time_ticks[time_slider])

    time_x = time_x[last_idx - analysis_dur + 1 : last_idx + 1]
    predicted_y = predicted_y[last_idx - analysis_dur + 1 : last_idx + 1]
    real_y = real_y[last_idx - analysis_dur + 1 : last_idx + 1]

    feat1 = feat1[last_idx - analysis_dur + 1 : last_idx + 1]
    feat2 = feat2[last_idx - analysis_dur + 1 : last_idx + 1]
    feat3 = feat3[last_idx - analysis_dur + 1 : last_idx + 1]
    feat4 = feat4[last_idx - analysis_dur + 1 : last_idx + 1]
    feat5 = feat5[last_idx - analysis_dur + 1 : last_idx + 1]

    risk_fig = make_subplots(specs=[[{"secondary_y": True}]])
    # fig.add_trace(go.Bar(x=time_x, y=predicted_y, name="예측 위험도", marker_color="indianred"))
    risk_fig.add_trace(
        go.Bar(
            x=time_x,
            y=real_y,
            name="실제 사고 빈도",
            marker_color="lightsalmon",
            hovertemplate="%{y}",
        ),
    )
    risk_fig.add_trace(
        go.Scatter(
            x=time_x,
            y=predicted_y,
            name="예측 위험도",
            marker_color="indianred",
            hovertemplate="%{y:.2f}",
        ),
        secondary_y=True,
    )

    risk_fig.update_layout(barmode="group")
    risk_fig.update_yaxes(secondary_y=False, title_text="사고 수")
    risk_fig.update_yaxes(
        secondary_y=True,
        title_text="위험도",
        range=[0.0, max(6.1, predicted_y.max() + 0.1)],
        showgrid=False,
    )
    risk_fig.add_vline(
        x=len(time_x) - 1,
        # x=0,
        line_width=2,
        line_color="red",
        opacity=0.2,
        annotation={"text": "기준 시점", "align": "left", "valign": "top"},
    )

    risk_fig.update_xaxes(type="category", tickformat="%s")
    risk_fig.update_layout(legend=dict(yanchor="top", y=-0.13, xanchor="right", x=0.95))
    risk_fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    feature_fig = make_subplots(specs=[[{"secondary_y": True}]])

    feature_fig.add_trace(
        go.Bar(
            x=time_x,
            y=feat1,
            name=feat_names[0],
            marker_color="lightsalmon",
            hovertemplate="%{y}",
        ),
    )
    feature_fig.add_trace(
        go.Bar(
            x=time_x,
            y=feat2,
            name=feat_names[1],
            marker_color="cadetblue",
            hovertemplate="%{y}",
        ),
    )
    feature_fig.add_trace(
        go.Bar(
            x=time_x,
            y=feat3,
            name=feat_names[2],
            marker_color="orchid",
            hovertemplate="%{y}",
        ),
    )
    feature_fig.add_trace(
        go.Bar(
            x=time_x,
            y=feat4,
            name=feat_names[3],
            marker_color="skyblue",
            hovertemplate="%{y}",
        ),
    )
    feature_fig.add_trace(
        go.Bar(
            x=time_x,
            y=feat5,
            name=feat_names[4],
            marker_color="darkorchid",
            hovertemplate="%{y}",
        ),
    )
    feature_fig.add_trace(
        go.Scatter(
            x=time_x,
            y=predicted_y,
            name="예측 위험도",
            marker_color="indianred",
            hovertemplate="%{y:.2f}",
        ),
        secondary_y=True,
    )
    feature_fig.add_vline(
        x=len(time_x) - 1,
        # x=0,
        line_width=2,
        line_color="red",
        opacity=0.2,
        annotation={"text": "기준 시점", "align": "left", "valign": "top"},
    )

    feature_fig.update_xaxes(type="category", tickformat="%s")
    feature_fig.update_layout(barmode="group")
    feature_fig.update_yaxes(secondary_y=False, title_text="특징 값")
    feature_fig.update_yaxes(
        secondary_y=True,
        title_text="위험도",
        range=[0.0, max(6.1, predicted_y.max() + 0.1)],
        showgrid=False,
    )
    feature_fig.update_layout(legend=dict(yanchor="top", y=-0.13, xanchor="right", x=0.95))
    feature_fig.update_layout(margin=dict(l=10, r=10, b=10, t=10))

    patch = Patch()
    patch["data"][1]["lat"] = [dff.iloc[0]["latitude"]]
    patch["data"][1]["lon"] = [dff.iloc[0]["longitude"]]

    return risk_fig, feature_fig, patch
    raise PreventUpdate


if __name__ == "__main__":
    app.run(debug=True)
