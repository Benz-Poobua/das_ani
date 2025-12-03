"""
:module: src/plot.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Plotting utilities for dispersion imaging (f–v panels),
          dispersion curve picks, and related DAS diagnostics.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go

from src.utils import convert_to_numpy

# Global Plotly style
PLOTLY_FONT_FAMILY = "Helvetica"

# 1. Matplotlib: f–v dispersion panel
# ===========================================================================
def plot_fv_panel(
    fv_panel,
    f_axis,
    v_axis,
    title="Dispersion Image (f–v panel)",
    cmap="viridis",
    figsize=(7, 5),
    save_path=None,
    show=True):
    """
    Plot dispersion image (frequency–velocity panel) using Matplotlib.

    :param fv_panel: Dispersion image of shape (n_vel, n_freq).
    :type fv_panel: numpy.ndarray or torch.Tensor
    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param v_axis: Phase-velocity axis in m/s, shape (n_vel,).
    :type v_axis: numpy.ndarray or torch.Tensor
    :param title: Figure title.
    :type title: str
    :param cmap: Matplotlib colormap name.
    :type cmap: str
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple[float, float]
    :param save_path: If provided, path to save the figure.
    :type save_path: str or None
    :param show: If True, call plt.show(), else close the figure.
    :type show: bool

    :return: Matplotlib (figure, axes) tuple.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    P = convert_to_numpy(fv_panel)
    f = convert_to_numpy(f_axis)
    vel = convert_to_numpy(v_axis)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        P,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[f.min(), f.max(), vel.min(), vel.max()],
    )

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_title(title)

    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Amplitude (normalized)")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

# 2. Matplotlib: f–v panel + pick curve overlay
# ===========================================================================
def plot_fv_with_pick(
    fv_panel,
    f_axis,
    v_axis,
    pick_curve,
    title="Dispersion with Picked Curve",
    cmap="viridis",
    figsize=(7, 5),
    save_path=None,
    show=True):
    """
    Plot dispersion image and overlay the picked dispersion curve using Matplotlib.

    :param fv_panel: Dispersion image of shape (n_vel, n_freq).
    :type fv_panel: numpy.ndarray or torch.Tensor
    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param v_axis: Phase-velocity axis in m/s, shape (n_vel,).
    :type v_axis: numpy.ndarray or torch.Tensor
    :param pick_curve: Picked phase velocities v(f), shape (n_freq,).
    :type pick_curve: numpy.ndarray or torch.Tensor
    :param title: Figure title.
    :type title: str
    :param cmap: Matplotlib colormap name.
    :type cmap: str
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple[float, float]
    :param save_path: If provided, path to save the figure.
    :type save_path: str or None
    :param show: If True, call plt.show(), else close the figure.
    :type show: bool

    :return: Matplotlib (figure, axes) tuple.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    P = convert_to_numpy(fv_panel)
    f = convert_to_numpy(f_axis)
    vel = convert_to_numpy(v_axis)
    pick = convert_to_numpy(pick_curve)

    fig, ax = plt.subplots(figsize=figsize)

    im = ax.imshow(
        P,
        aspect="auto",
        origin="lower",
        cmap=cmap,
        extent=[f.min(), f.max(), vel.min(), vel.max()],
    )

    ax.plot(f, pick, "r-", linewidth=2.5, label="Picked dispersion")
    ax.legend()

    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_title(title)

    plt.colorbar(im, ax=ax, label="Amplitude (normalized)")

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

# 3. Matplotlib: v(f) pick curve only
def plot_pick_curve(
    f_axis,
    pick_curve,
    title="Dispersion Curve",
    figsize=(6, 4),
    save_path=None,
    show=True):
    """
    Plot dispersion curve v(f) using Matplotlib.

    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param pick_curve: Picked phase velocities v(f), shape (n_freq,).
    :type pick_curve: numpy.ndarray or torch.Tensor
    :param title: Figure title.
    :type title: str
    :param figsize: Figure size (width, height) in inches.
    :type figsize: tuple[float, float]
    :param save_path: If provided, path to save the figure.
    :type save_path: str or None
    :param show: If True, call plt.show(), else close the figure.
    :type show: bool

    :return: Matplotlib (figure, axes) tuple.
    :rtype: tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]
    """
    f = convert_to_numpy(f_axis)
    pick = convert_to_numpy(pick_curve)

    fig, ax = plt.subplots(figsize=figsize)

    ax.plot(f, pick, "k-", linewidth=2)
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase velocity (m/s)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches="tight")

    if show:
        plt.show()
    else:
        plt.close()

    return fig, ax

# 4. Plotly: f–v dispersion panel (interactive)
# ===========================================================================
def plot_fv_panel_plotly(
    fv_panel,
    f_axis,
    v_axis,
    fmin=None,
    fmax=None,
    vmin=None,
    vmax=None,
    gauge_len=None,
    array_len=None,
    cmap="Jet",
    title="Dispersion Image (f–v panel)",
    width=800,
    height=600,
    fontsize=14,
    show=True,
    savepath=None):
    """
    Plot dispersion image (f–v panel) using Plotly with optional
    v = 2 L f guide lines.

    :param fv_panel: Dispersion image of shape (n_vel, n_freq).
    :type fv_panel: numpy.ndarray or torch.Tensor
    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param v_axis: Phase-velocity axis in m/s, shape (n_vel,).
    :type v_axis: numpy.ndarray or torch.Tensor
    :param fmin: Minimum frequency for display (Hz).
    :type fmin: float or None
    :param fmax: Maximum frequency for display (Hz).
    :type fmax: float or None
    :param vmin: Minimum velocity for display (m/s).
    :type vmin: float or None
    :param vmax: Maximum velocity for display (m/s).
    :type vmax: float or None
    :param gauge_len: If provided, plot v = 2 * gauge_len * f.
    :type gauge_len: float or None
    :param array_len: If provided, plot v = 2 * array_len * f.
    :type array_len: float or None
    :param cmap: Plotly colormap name ('Jet', 'Viridis', etc.).
    :type cmap: str
    :param title: Plot title.
    :type title: str
    :param width: Figure width in pixels.
    :type width: int
    :param height: Figure height in pixels.
    :type height: int
    :param fontsize: Base font size.
    :type fontsize: int
    :param show: If True, display in notebook/browser.
    :type show: bool
    :param savepath: If provided, save as HTML (or static image if supported).
    :type savepath: str or None

    :return: Plotly Figure object.
    :rtype: plotly.graph_objects.Figure
    """
    fv = convert_to_numpy(fv_panel)
    f = convert_to_numpy(f_axis)
    v = convert_to_numpy(v_axis)

    # Frequency mask
    if fmin is not None or fmax is not None:
        fmin_eff = fmin if fmin is not None else f[0]
        fmax_eff = fmax if fmax is not None else f[-1]
        f_mask = (f >= fmin_eff) & (f <= fmax_eff)
        f = f[f_mask]
        fv = fv[:, f_mask]

    # Velocity mask
    if vmin is not None or vmax is not None:
        vmin_eff = vmin if vmin is not None else v[0]
        vmax_eff = vmax if vmax is not None else v[-1]
        v_mask = (v >= vmin_eff) & (v <= vmax_eff)
        v = v[v_mask]
        fv = fv[v_mask, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=fv,
            x=f,
            y=v,
            colorscale=cmap,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Amp.",
                    side="right",
                    font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                ),
                tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                thickness=10,
                xpad=5,
            ),
        )
    )

    # Optional guide lines
    if gauge_len is not None:
        v_line = 2.0 * gauge_len * f
        fig.add_trace(
            go.Scatter(
                x=f,
                y=v_line,
                mode="lines",
                line=dict(color="white", width=2, dash="dash"),
                name="v = 2·L_gauge·f",
            )
        )

    if array_len is not None:
        v_line2 = 2.0 * array_len * f
        fig.add_trace(
            go.Scatter(
                x=f,
                y=v_line2,
                mode="lines",
                line=dict(color="white", width=2, dash="dot"),
                name="v = 2·L_array·f",
            )
        )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Frequency (Hz)",
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            showgrid=True,
        ),
        yaxis=dict(
            title="Phase velocity (m/s)",
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            showgrid=True,
            range=[vmin, vmax] if (vmin is not None and vmax is not None) else None,
        ),
        width=width,
        height=height,
        font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
    )

    if savepath:
        if savepath.endswith(".html"):
            fig.write_html(savepath)
        else:
            fig.write_image(savepath, width=width, height=height, scale=2)

    if show:
        fig.show()

    return fig

# 5. Plotly: f–v + pick overlay
# ===========================================================================
def plot_fv_with_pick_plotly(
    fv_panel,
    f_axis,
    v_axis,
    pick_curve,
    fmin=None,
    fmax=None,
    vmin=None,
    vmax=None,
    cmap="Jet",
    title="Dispersion with Picked Curve",
    width=800,
    height=600,
    fontsize=14,
    show=True,
    savepath=None):
    """
    Plot dispersion image and overlay v(f) pick using Plotly.

    :param fv_panel: Dispersion image of shape (n_vel, n_freq).
    :type fv_panel: numpy.ndarray or torch.Tensor
    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param v_axis: Phase-velocity axis in m/s, shape (n_vel,).
    :type v_axis: numpy.ndarray or torch.Tensor
    :param pick_curve: Picked phase velocities v(f), shape (n_freq,).
    :type pick_curve: numpy.ndarray or torch.Tensor
    :param fmin: Minimum frequency for display (Hz).
    :type fmin: float or None
    :param fmax: Maximum frequency for display (Hz).
    :type fmax: float or None
    :param vmin: Minimum velocity for display (m/s).
    :type vmin: float or None
    :param vmax: Maximum velocity for display (m/s).
    :type vmax: float or None
    :param cmap: Plotly colormap name.
    :type cmap: str
    :param title: Plot title.
    :type title: str
    :param width: Figure width in pixels.
    :type width: int
    :param height: Figure height in pixels.
    :type height: int
    :param fontsize: Base font size.
    :type fontsize: int
    :param show: If True, display in notebook/browser.
    :type show: bool
    :param savepath: If provided, path to save as HTML or image.
    :type savepath: str or None

    :return: Plotly Figure object.
    :rtype: plotly.graph_objects.Figure
    """
    fv = convert_to_numpy(fv_panel)
    f = convert_to_numpy(f_axis)
    v = convert_to_numpy(v_axis)
    pick = convert_to_numpy(pick_curve)

    # Frequency mask for both fv and pick
    if fmin is not None or fmax is not None:
        fmin_eff = fmin if fmin is not None else f[0]
        fmax_eff = fmax if fmax is not None else f[-1]
        f_mask = (f >= fmin_eff) & (f <= fmax_eff)
        f = f[f_mask]
        fv = fv[:, f_mask]
        pick = pick[f_mask]

    # Velocity mask
    if vmin is not None or vmax is not None:
        vmin_eff = vmin if vmin is not None else v[0]
        vmax_eff = vmax if vmax is not None else v[-1]
        v_mask = (v >= vmin_eff) & (v <= vmax_eff)
        v = v[v_mask]
        fv = fv[v_mask, :]

    fig = go.Figure(
        data=go.Heatmap(
            z=fv,
            x=f,
            y=v,
            colorscale=cmap,
            showscale=True,
            colorbar=dict(
                title=dict(
                    text="Amp.",
                    side="right",
                    font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                ),
                tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                thickness=10,
                xpad=5,
            ),
        )
    )

    fig.add_trace(
        go.Scatter(
            x=f,
            y=pick,
            mode="lines",
            line=dict(color="red", width=2.5),
            name="Picked dispersion",
        )
    )

    fig.update_layout(
        title=title,
        xaxis=dict(
            title="Frequency (Hz)",
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            showgrid=True,
        ),
        yaxis=dict(
            title="Phase velocity (m/s)",
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            showgrid=True,
            range=[vmin, vmax] if (vmin is not None and vmax is not None) else None,
        ),
        width=width,
        height=height,
        font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
    )

    if savepath:
        if savepath.endswith(".html"):
            fig.write_html(savepath)
        else:
            fig.write_image(savepath, width=width, height=height, scale=2)

    if show:
        fig.show()

    return fig

# 6. Plotly: v(f) pick curve only
# ===========================================================================
def plot_pick_curve_plotly(
    f_axis,
    pick_curve,
    title="Dispersion Curve",
    xlabel="Frequency (Hz)",
    ylabel="Phase velocity (m/s)",
    width=800,
    height=500,
    line_color="blue",
    line_width=2,
    font_size=16,
    xlim=None,
    ylim=None,
    show=True,
    savepath=None):
    """
    Plot dispersion curve v(f) using Plotly.

    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param pick_curve: Picked phase velocities v(f), shape (n_freq,).
    :type pick_curve: numpy.ndarray or torch.Tensor
    :param title: Plot title.
    :type title: str
    :param xlabel: Label for x-axis.
    :type xlabel: str
    :param ylabel: Label for y-axis.
    :type ylabel: str
    :param width: Figure width in pixels.
    :type width: int
    :param height: Figure height in pixels.
    :type height: int
    :param line_color: Line color.
    :type line_color: str
    :param line_width: Line width (pixels).
    :type line_width: int
    :param font_size: Base font size.
    :type font_size: int
    :param xlim: Optional (xmin, xmax) for x-axis.
    :type xlim: tuple[float, float] or None
    :param ylim: Optional (ymin, ymax) for y-axis.
    :type ylim: tuple[float, float] or None
    :param show: If True, display plot.
    :type show: bool
    :param savepath: If provided, save as HTML or image.
    :type savepath: str or None

    :return: Plotly Figure object.
    :rtype: plotly.graph_objects.Figure
    """
    f = convert_to_numpy(f_axis)
    pick = convert_to_numpy(pick_curve)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=f,
            y=pick,
            mode="lines",
            name="Dispersion curve",
            line=dict(color=line_color, width=line_width),
        )
    )

    layout_kwargs = dict(
        title=title,
        xaxis=dict(title=xlabel),
        yaxis=dict(title=ylabel),
        width=width,
        height=height,
        template="plotly_white",
        font=dict(size=font_size, family=PLOTLY_FONT_FAMILY),
    )

    if xlim is not None:
        layout_kwargs["xaxis"]["range"] = list(xlim)
    if ylim is not None:
        layout_kwargs["yaxis"]["range"] = list(ylim)

    fig.update_layout(**layout_kwargs)

    if savepath:
        if savepath.endswith(".html"):
            fig.write_html(savepath)
        else:
            fig.write_image(savepath, width=width, height=height, scale=2)

    if show:
        fig.show()

    return fig

# 7. Plotly: animation over days for f–v panels
# ===========================================================================
def animate_fv_over_days_plotly(
    fv_stack,
    f_axis,
    v_axis,
    day_labels=None,
    cmap="Jet",
    title="Dispersion evolution over days",
    width=800,
    height=600,
    fontsize=14):
    """
    Animate a sequence of f–v panels over days using Plotly.

    :param fv_stack: 3D stack of dispersion images,
                     shape (n_days, n_vel, n_freq).
    :type fv_stack: numpy.ndarray
    :param f_axis: Frequency axis in Hz, shape (n_freq,).
    :type f_axis: numpy.ndarray or torch.Tensor
    :param v_axis: Phase-velocity axis in m/s, shape (n_vel,).
    :type v_axis: numpy.ndarray or torch.Tensor
    :param day_labels: Labels for each frame (e.g., date strings).
                       If None, use integer indices.
    :type day_labels: list[str] or None
    :param cmap: Plotly colormap name.
    :type cmap: str
    :param title: Base plot title.
    :type title: str
    :param width: Figure width in pixels.
    :type width: int
    :param height: Figure height in pixels.
    :type height: int
    :param fontsize: Base font size.
    :type fontsize: int

    :return: Plotly Figure object with slider/animation.
    :rtype: plotly.graph_objects.Figure
    """
    fv_stack = convert_to_numpy(fv_stack)
    f = convert_to_numpy(f_axis)
    v = convert_to_numpy(v_axis)

    n_days, n_vel, n_freq = fv_stack.shape
    if day_labels is None:
        day_labels = [f"day_{i}" for i in range(n_days)]
    if len(day_labels) != n_days:
        raise ValueError("day_labels length must match first dimension of fv_stack.")

    # Initial frame
    heatmap = go.Heatmap(
        z=fv_stack[0],
        x=f,
        y=v,
        colorscale=cmap,
        showscale=True,
        colorbar=dict(
            title=dict(
                text="Amp.",
                side="right",
                font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            ),
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            thickness=10,
            xpad=5,
        ),
    )

    frames = []
    for i in range(n_days):
        frames.append(
            go.Frame(
                data=[
                    go.Heatmap(
                        z=fv_stack[i],
                        x=f,
                        y=v,
                        colorscale=cmap,
                        zmin=fv_stack.min(),
                        zmax=fv_stack.max(),
                        showscale=True,
                    )
                ],
                name=str(i),
            )
        )

    # Slider steps
    steps = []
    for i, label in enumerate(day_labels):
        steps.append(
            dict(
                method="animate",
                args=[[str(i)], {"frame": {"duration": 500, "redraw": True}, "mode": "immediate"}],
                label=label,
            )
        )

    sliders = [
        dict(
            steps=steps,
            transition={"duration": 0},
            x=0.2,
            xanchor="left",
            y=0,
            yanchor="top",
            currentvalue=dict(
                font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                prefix="Day: ",
                visible=True,
                xanchor="right",
            ),
            font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            len=0.8,
        )
    ]

    buttons = [
        dict(
            label="Play",
            method="animate",
            args=[
                None,
                {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "mode": "immediate"},
            ],
        ),
        dict(
            label="Stop",
            method="animate",
            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        ),
    ]

    layout = go.Layout(
        title=title,
        xaxis=dict(
            title="Frequency (Hz)",
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            showgrid=True,
        ),
        yaxis=dict(
            title="Phase velocity (m/s)",
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            showgrid=True,
        ),
        width=width,
        height=height,
        font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                showactive=False,
                x=0.0,
                y=-0.05,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            )
        ],
        sliders=sliders,
    )

    fig = go.Figure(data=[heatmap], layout=layout, frames=frames)
    fig.show()

    return fig

# 8. Other generic Plotly utilities
# ===========================================================================
def plot_data_animation_plotly(
    data,
    x,
    t,
    pclip=99.9,
    vmin=None,
    vmax=None,
    cmap="gray",
    y_ticks_interval=1,
    fontsize=18,
    width=1000,
    height=800,
    title=""):
    """
    Plot 3D wavefield data as an animation over sources using Plotly.

    :param data: 3D array, shape (n_src, n_x, n_t).
    :type data: numpy.ndarray or list
    :param x: Spatial axis (distance), length n_x.
    :type x: numpy.ndarray
    :param t: Time axis, length n_t.
    :type t: numpy.ndarray
    :param pclip: Percentile for symmetric clipping if vmin/vmax not provided.
    :type pclip: float
    :param vmin: Minimum amplitude for color scale.
    :type vmin: float or None
    :param vmax: Maximum amplitude for color scale.
    :type vmax: float or None
    :param cmap: Plotly colormap name.
    :type cmap: str
    :param y_ticks_interval: Interval for horizontal grid lines in time units.
    :type y_ticks_interval: float
    :param fontsize: Base font size.
    :type fontsize: int
    :param width: Figure width in pixels.
    :type width: int
    :param height: Figure height in pixels.
    :type height: int
    :param title: Plot title.
    :type title: str

    :return: Plotly Figure with animation frames.
    :rtype: plotly.graph_objects.Figure
    """
    if isinstance(data, list):
        data = np.array(data)
    data = np.squeeze(data)
    nsrc, nx, nt = data.shape

    if data.ndim != 3:
        raise ValueError("data must be 3D (n_src, n_x, n_t)")

    if len(x) != nx:
        raise ValueError("x must match nx")
    if len(t) != nt:
        raise ValueError("t must match nt")

    if vmin is None or vmax is None:
        clip_max = np.percentile(data, pclip)
        clip_min = -clip_max
    else:
        clip_min, clip_max = vmin, vmax

    frames = [
        go.Frame(
            data=[
                go.Heatmap(
                    z=data[i].T,
                    x=x,
                    y=t,
                    zmin=clip_min,
                    zmax=clip_max,
                    colorscale=cmap,
                    reversescale=True,
                    showscale=True,
                )
            ],
            name=str(i),
        )
        for i in range(nsrc)
    ]

    heatmap = go.Heatmap(
        z=data[0].T,
        x=x,
        y=t,
        zmin=clip_min,
        zmax=clip_max,
        colorscale=cmap,
        colorbar=dict(
            title=dict(text="Amp.", side="right", font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY)),
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            thickness=10,
            xpad=5,
        ),
        reversescale=True,
        showscale=True,
    )

    # Grid lines in time
    grid_lines = []
    start_tick = np.ceil(t[0] / y_ticks_interval + 1) * y_ticks_interval
    y_ticks = np.arange(start_tick, t[-1], y_ticks_interval)
    for yi in y_ticks:
        grid_lines.append(
            go.Scatter(
                x=[x[0], x[-1]],
                y=[yi, yi],
                mode="lines",
                line=dict(color="lightgray", width=1, dash="dash"),
                hoverinfo="skip",
                showlegend=False,
            )
        )

    buttons = [
        dict(
            label="Play",
            method="animate",
            args=[None, {"frame": {"duration": 1000, "redraw": True}, "fromcurrent": True, "mode": "immediate"}],
        ),
        dict(
            label="Stop",
            method="animate",
            args=[[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}],
        ),
    ]

    layout = go.Layout(
        font=dict(family=PLOTLY_FONT_FAMILY, size=fontsize),
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY)),
        xaxis=dict(
            title="Distance (m)",
            side="top",
            showticklabels=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1,
            tickcolor="black",
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
        ),
        yaxis=dict(
            title="Time (s)",
            autorange="reversed",
            showticklabels=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1,
            tickcolor="black",
            titlefont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            tickfont=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
        ),
        width=width,
        height=height,
        updatemenus=[
            dict(
                type="buttons",
                direction="down",
                showactive=False,
                x=0.0,
                y=-0.05,
                xanchor="left",
                yanchor="top",
                buttons=buttons,
                font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
            )
        ],
        sliders=[
            dict(
                steps=[
                    dict(
                        method="animate",
                        args=[[str(i)], {"frame": {"duration": 1000, "redraw": True}, "mode": "immediate"}],
                        label=str(i),
                    )
                    for i in range(nsrc)
                ],
                transition={"duration": 0},
                x=0.2,
                xanchor="left",
                y=0,
                yanchor="top",
                currentvalue=dict(
                    font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                    prefix="Source: ",
                    visible=True,
                    xanchor="right",
                ),
                font=dict(size=fontsize, family=PLOTLY_FONT_FAMILY),
                len=0.8,
            )
        ],
    )

    fig = go.Figure(data=[heatmap] + grid_lines, layout=layout, frames=frames)
    return fig


def plot_power_spectrum_plotly(
    freqs,
    power_db,
    dx=1.0,
    vmin=None,
    vmax=None,
    fmin=None,
    fmax=None,
    xtick_interval=500,
    ytick_interval=10):
    """
    Plot power spectrum (channel vs frequency) using Plotly.

    :param freqs: Frequency axis in Hz, shape (n_freq,).
    :type freqs: numpy.ndarray
    :param power_db: Power spectral density in dB, shape (n_ch, n_freq).
    :type power_db: numpy.ndarray
    :param dx: Channel spacing in meters.
    :type dx: float
    :param vmin: Minimum color scale (dB after shifting to min=0).
    :type vmin: float or None
    :param vmax: Maximum color scale (dB after shifting to min=0).
    :type vmax: float or None
    :param fmin: Minimum frequency to display (Hz).
    :type fmin: float or None
    :param fmax: Maximum frequency to display (Hz).
    :type fmax: float or None
    :param xtick_interval: Tick spacing along channel-distance axis (m).
    :type xtick_interval: float
    :param ytick_interval: Tick spacing along frequency axis (Hz).
    :type ytick_interval: float

    :return: Plotly Figure object.
    :rtype: plotly.graph_objects.Figure
    """
    zmin = 0.0 if vmin is None else vmin
    zmax = power_db.max() if vmax is None else vmax

    nch, nfreq = power_db.shape
    ch_axis = np.arange(nch) * dx

    fmin_eff = freqs[0] if fmin is None else fmin
    fmax_eff = freqs[-1] if fmax is None else fmax
    freq_mask = (freqs >= fmin_eff) & (freqs <= fmax_eff)
    freqs_plot = freqs[freq_mask]
    power_plot = power_db[:, freq_mask]

    xticks = np.arange(0, ch_axis[-1] + 1, xtick_interval)
    yticks = np.arange(fmin_eff, fmax_eff + 1, ytick_interval)

    fig = go.Figure(
        data=go.Heatmap(
            z=power_plot.T,
            x=ch_axis,
            y=freqs_plot,
            zmin=zmin,
            zmax=zmax,
            showscale=True,
            colorscale="Jet",
            colorbar=dict(title="Power (dB)"),
        )
    )

    fig.update_layout(
        title="Power Spectrum",
        xaxis=dict(
            title="Channel Distance (m)",
            tickmode="array",
            tickvals=xticks,
            ticktext=[f"{x:.0f}" for x in xticks],
            showgrid=True,
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
        ),
        yaxis=dict(
            title="Frequency (Hz)",
            tickmode="array",
            tickvals=yticks,
            ticktext=[f"{y:.0f}" for y in yticks],
            showgrid=True,
            showline=True,
            ticks="outside",
            ticklen=5,
            tickwidth=1.5,
            mirror=True,
            range=[fmin_eff, fmax_eff],
        ),
        width=900,
        height=600,
    )

    fig.show()
    return fig