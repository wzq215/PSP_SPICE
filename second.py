import plotly.graph_objects as go

fig = go.Figure()

# 添加左侧纵坐标轴数据
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=[10, 20, 30, 40, 50],
    name="左侧数据"
))

# 添加右侧纵坐标轴数据
y2 = [50, 40, 30, 20, 10] * 3
fig.add_trace(go.Scatter(
    x=[1, 2, 3, 4, 5],
    y=y2  # [150, 120, 90, 60, 30],
name = "右侧数据",
yaxis = "y2"  # 指定使用右侧纵坐标轴
))

# 设置布局
fig.update_layout(
    yaxis=dict(title="左侧纵坐标轴标题"),
    yaxis2=dict(title="右侧纵坐标轴标题", overlaying="y", side="right"),  # 设置右侧纵坐标轴标题
    legend=dict(x=0, y=1)  # 设置图例位置
)

# 保存为HTML文件
fig.write_html("plot.html")
