import numpy as np
import plotly.figure_factory as ff
from skimage import measure
import plotly as py
import plotly.graph_objs as go

def plot_iso_surface(fcn_grid, iso_level, n_samples, grid_max, title, color, display_plot):
    spacing = 2*grid_max/n_samples
    vertices, simplices, _, _ = measure.marching_cubes_lewiner(fcn_grid, iso_level, spacing=(spacing, spacing, spacing))
    
    x,y,z = zip(*vertices) 
    x = np.array(x) - grid_max
    y = np.array(y) - grid_max
    z = np.array(z) - grid_max
    
    hover_text = generate_hover_text(x, y, z, 'Weight on Stock A', 'Weight on Stock B', 'Weight on Stock C')

    fig = ff.create_trisurf(x=x,
                            y=y, 
                            z=z, 
                            plot_edges=False,
                            show_colorbar=False,
                            colormap=color,
                            simplices=simplices)
    fig['data'][0].update(opacity=0.3, hoverinfo='none')
    
    trace = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode='markers',
        marker=dict(
            size=6,
            opacity=0.0001,
            color='#BFB1A8', 
        ),
        text = hover_text.flatten(),
        hoverinfo = 'text',
        showlegend=False
    )

    layout = create_standard_layout(title, 'Weight on')

    data = [fig.data[0], trace]

    fig = go.Figure(data=data, layout=layout)
    if display_plot:
        py.offline.iplot(fig)
    return data


def generate_hover_text(x_values, y_values, z_values, x_label, y_label, z_label, **kwargs):
    fcn_label = ''
    sig_figs = 3
    
    for key in kwargs:
        if key == 'fcn_values':
            fcn_values = kwargs[key]
        if key == 'fcn_label':
            fcn_label = kwargs[key]
        if key == 'sig_figs':
            sig_figs = kwargs[key]
            
    sig_figs = '{:.'+ str(sig_figs) +'f}'       
            
    float_to_str = np.vectorize(sig_figs.format)
    
    padding_len = np.full(4, max(len(x_label), len(y_label), len(z_label), len(fcn_label))) - \
                  [len(x_label), len(y_label), len(z_label), len(fcn_label)]

    # Additional padding added to ticker and date to align
    hover_text = x_label + ':  ' + padding_len[0] * ' ' + float_to_str(x_values).astype(object) + '<br>' + \
                 y_label + ':  ' + padding_len[1] * ' ' + float_to_str(y_values).astype(object) + '<br>' + \
                 z_label + ': ' + padding_len[2] * ' ' + float_to_str(z_values).astype(object) 
            
    if 'fcn_values' in kwargs:
        hover_text = hover_text + '<br>' + \
                 fcn_label + ': ' + padding_len[3] * ' ' + float_to_str(fcn_values).astype(object)

    return hover_text


def create_standard_layout(title, axis_text):
    layout = go.Layout(
    title=title,
    margin=dict(
        l=50,
        r=50,
        b=50,
        t=50
    ),
    scene=go.Scene(
        xaxis=dict(
            title=axis_text + ' Stock A',
            zeroline=True,
            titlefont=dict(
                size=14,
                color='#7f7f7f'
            )
        ),
        yaxis=dict(
            title=axis_text + ' Stock B',
            zeroline=True,
            titlefont=dict(
                size=14,
                color='#7f7f7f'
            )
        ),
        zaxis=dict(
            title=axis_text + ' Stock C',
            zeroline=True,
            titlefont=dict(
                size=14,
                color='#7f7f7f'
                    )
                )
            )
        )
    return layout

def evaluate_fcn_on_grid(grid_max, n_samples, fcn):
    axis = np.linspace(-grid_max, grid_max, n_samples)
    xv, yv, zv = np.meshgrid(axis, axis, axis, indexing='ij')
    spacing = 2*grid_max/n_samples
    
    # pre-allocate fcn_grid
    tmp_out = fcn(np.array([1,2,3]))
    if type(tmp_out) == np.ndarray:
        fcn_grid = np.full([n_samples, n_samples, n_samples, len(tmp_out)], np.nan) 
    else:
        fcn_grid = np.full([n_samples, n_samples, n_samples], np.nan)
    
    for i in range(n_samples):
        for j in range(n_samples):
            for k in range(n_samples):
                x = np.array([xv[i,j,k], yv[i,j,k], zv[i,j,k]])
                fcn_out = fcn(x)
                if type(fcn_out) == np.ndarray:
                    for e in range(len(fcn_out)):
                        fcn_grid[i, j, k, e] = fcn_out[e]
                else:
                    fcn_grid[i, j, k] = fcn_out
    
    return fcn_grid, spacing, xv, yv, zv