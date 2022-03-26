import plotly.graph_objects as go
import plotly.offline as py

fig= go.Figure(data=go.Isosurface(
    x=[0,0,0,0,1,1,1,1],
    y=[1,0,1,0,1,0,1,0],
    z=[1,1,0,0,1,1,0,0],
    value=[1,2,3,4,5,6,7,8],
    isomin=1,
    isomax=8,
    surface_count=1,
    colorbar_nticks=1,
    caps=dict(x_show=False, y_show=False)
))

# fig.show()
py.plot(fig,filename='tesst.html',image='png')