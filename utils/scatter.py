import matplotlib.pyplot as plt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas

def scatter_by_mass(mass_values, data, x_var='ALG', y_var='PVA', z_var='GUM', 
                    y_col='Kale afw (g/cm2)', ncols=2, nrows=1, 
                    vmin=0.0, vmax=1, marker='o', linewidths=0.5, s=50, alpha=1, 
                    elev=25, azim=120, cmap='viridis', edgecolor='face', 
                    show_cbar=True, cbar_label='Fresh Wright (mg/cm^2)', 
                    zproj=False, show_cbar_ticks=True, cbar_tick_position=None, 
                    cbar_ticks=None, show_ticks=True, show_labels=True,
                    cbar_pad=0.1, label_pad=8, cbar_tick_label_rotation=90,
                    figsize=None, rasterized=False,
                    cbar_shrink=0.8, cbar_aspect=20):
    """
    Create 3D scatter plots for different mass loadings and y variables.
    
    Parameters
    ----------
    y_col : str or list of str
        Single target variable or list of target variables to plot
    ncols : int
        Number of columns (typically number of mass values)
    nrows : int
        Number of rows (automatically calculated if y_col is a list)
    cbar_label : str or list of str
        Colorbar label(s). If list, must match length of y_col list.
        If single string and y_col is list, uses y_col names as labels.
    
    Notes
    -----
    When y_col is a list:
    - Each row shows a different y variable
    - Each column shows a different mass loading
    - Each subplot has its own colorbar on the right
    """
    # Convert y_col to list for uniform handling
    y_cols = [y_col] if isinstance(y_col, str) else y_col
    n_y_vars = len(y_cols)
    
    # Handle colorbar labels
    if isinstance(cbar_label, str):
        cbar_labels = [y_col_name for y_col_name in y_cols] if n_y_vars > 1 else [cbar_label]
    else:
        cbar_labels = cbar_label
    
    # Adjust figure layout: rows = y variables, cols = mass values
    actual_nrows = n_y_vars
    actual_ncols = len(mass_values)
    
    fig = plt.figure(figsize=figsize if figsize is not None else (actual_ncols * 8, actual_nrows * 5))
    
    subplot_idx = 1
    for y_idx, y_col_name in enumerate(y_cols):
        for mass_idx, mass in enumerate(mass_values):
            ax = fig.add_subplot(actual_nrows, actual_ncols, subplot_idx, projection='3d')
            
            subset = data.loc[data['MassLoading'] == mass]
            x, y, z, c = subset[x_var], subset[y_var], subset[z_var], subset[y_col_name]
            
            sc = ax.scatter(x, y, z, c=c, cmap=cmap, s=s, vmin=vmin, vmax=vmax,
                           marker=marker, linewidths=linewidths, edgecolor=edgecolor,
                           alpha=alpha, zorder=2, rasterized=rasterized)
            
            # Show colorbar on the right of each subplot
            if show_cbar:
                cbar = plt.colorbar(sc, ax=ax, shrink=cbar_shrink, pad=cbar_pad,
                                    aspect=cbar_aspect, alpha=1.0)
                cbar.solids.set_alpha(1.0)
                cbar.set_label(cbar_labels[y_idx], size=14)

                if not show_cbar_ticks:
                    cbar.ax.set_yticklabels([])
                    cbar.ax.tick_params(length=0)
                else:
                    cbar.ax.tick_params(labelsize=12)
                    if cbar_ticks is not None:
                        cbar.ax.set_yticks(cbar_tick_position)
                        cbar.ax.set_yticklabels(cbar_ticks, fontsize=14,
                                                rotation=cbar_tick_label_rotation,
                                                va='center')
            
            if zproj:
                for i in range(len(subset)):
                    ax.plot([x.iloc[i], x.iloc[i]], [y.iloc[i], y.iloc[i]], 
                           [0, z.iloc[i]], color='gray', linewidth=1.0, linestyle='--')
            
            # Update title to show both y variable and mass loading
            title = f'{y_col_name}\nMass Loading = {round(mass)} (mg/mL)'
            plt.title(title, fontsize=16)
            
            ax.view_init(elev=elev, azim=azim)
            ax.set_xlim([0, 1])
            ax.set_ylim([0, 1])
            ax.set_zlim([0, 1])
            
            if show_ticks:
                ax.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             [0, 20, 40, 60, 80, 100], fontsize=13)
                ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             [0, 20, 40, 60, 80, 100], fontsize=13)
                ax.set_zticks([0, 0.2, 0.4, 0.6, 0.8, 1.0],
                             [0, 20, 40, 60, 80, 100], fontsize=13)
            else:
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_zticks([])
                ax.tick_params(axis='x', which='both', bottom=False, top=False, 
                              labelbottom=False)
                ax.tick_params(axis='y', which='both', left=False, right=False, 
                              labelleft=False)
                ax.tick_params(axis='z', which='both', left=False, right=False, 
                              labelleft=False)
            
            if show_labels:
                ax.set_xlabel(f'{x_var}', fontsize=14, labelpad=label_pad)
                ax.set_ylabel(f'{y_var}', fontsize=14, labelpad=label_pad)
                ax.set_zlabel(f'{z_var}', fontsize=14, labelpad=label_pad, rotation=90)
            else:
                ax.set_xlabel('')
                ax.set_ylabel('')
                ax.set_zlabel('')
            
            subplot_idx += 1
    
    return sc

def scatter_by_mass_go(mass_values, data, x_var='ALG', y_var='PVA', z_var='GUM',
                    y_col='Kale afw (g/cm2)', ncols=2, nrows=1, vmin=0.0, vmax=1,
                    marker_size=8, cmap='Viridis', show_cbar=True,
                    cbar_label='Fresh Weight (mg/cm^2)', zproj=False,
                    show_ticks=True, show_labels=True):

    # Create subplots
    fig = make_subplots(
        rows=nrows, cols=ncols,
        subplot_titles=[f'{y_col}<br>Mass Loading = {round(mass)} (mg/mL)' 
                        for mass in mass_values],
        specs=[[{'type': 'scatter3d'} for _ in range(ncols)] for _ in range(nrows)],
        horizontal_spacing=0.025,
        vertical_spacing=0.1
    )
    
    for idx, mass in enumerate(mass_values):
        row = idx // ncols + 1
        col = idx % ncols + 1
        
        subset = data.loc[data['MassLoading'] == mass]
        x, y, z, c = subset[x_var], subset[y_var], subset[z_var], subset[y_col]
        
        # Main scatter plot
        scatter = go.Scatter3d(
            x=x, y=y, z=z,
            mode='markers',
            marker=dict(
                size=marker_size,
                color=c,
                colorscale=cmap,
                cmin=vmin,
                cmax=vmax,
                colorbar=dict(
                    title=cbar_label,
                    tickfont=dict(size=12),
                    len=0.8,
                    x=1.02
                ) if show_cbar and idx == len(mass_values) - 1 else None,
                showscale=show_cbar and idx == len(mass_values) - 1
            ),
            showlegend=False
        )
        fig.add_trace(scatter, row=row, col=col)
        
        # Add projections if requested as a single trace
        if zproj:
            proj_x, proj_y, proj_z = [], [], []
            # Use .iloc for robust integer-based indexing
            for i in range(len(x)): 
                proj_x.extend([x.iloc[i], x.iloc[i], None])
                proj_y.extend([y.iloc[i], y.iloc[i], None])
                proj_z.extend([0, z.iloc[i], None]) # Line from z=0 to z_point
            
            fig.add_trace(
                go.Scatter3d(
                    x=proj_x,
                    y=proj_y,
                    z=proj_z,
                    mode='lines',
                    line=dict(color='gray', width=2, dash='dash'),
                    showlegend=False
                ),
                row=row, col=col
            )
        
        # Define axis settings
        axis_settings = dict(
            range=[0, 1], # Hard-coded assumption
            showticklabels=show_ticks,
            tickvals=[0, 0.2, 0.4, 0.6, 0.8, 1.0] if show_ticks else [],
            ticktext=['0', '20', '40', '60', '80', '100'] if show_ticks else [],
            tickfont=dict(size=14)
        )
        
        # Use fig.update_scenes to target the specific subplot's 3D scene
        fig.update_scenes(
            xaxis=dict(**axis_settings, title=dict(text=x_var if show_labels else '', font=dict(size=14))),
            yaxis=dict(**axis_settings, title=dict(text=y_var if show_labels else '', font=dict(size=14))),
            zaxis=dict(**axis_settings, title=dict(text=z_var if show_labels else '', font=dict(size=14))),
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.2)
            ),
            # This selector is crucial
            row=row, col=col 
        )
    
    # Update layout
    fig.update_layout(
        height=nrows * 500,
        width=ncols * 800,
        showlegend=False
    )
    
    return fig