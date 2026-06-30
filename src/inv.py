"""
:module: src/inv.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Inversion utilities
"""
import os
import contextlib
import numpy as np
from typing import Any, Literal, List, Dict, Tuple, Optional

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.axes import Axes
from matplotlib.ticker import ScalarFormatter
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib.lines import Line2D
from matplotlib.animation import FuncAnimation

from IPython.display import HTML, display
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.ndimage import gaussian_filter

from evodcinv import EarthModel, Layer, Curve
from disba import PhaseSensitivity, surf96, depthplot
from disba._common import ifunc

np.Inf = np.inf

# Global Publication Typography Settings
mpl.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans'],
    'axes.labelsize': 14,
    'axes.titlesize': 16,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.dpi': 150,
    'axes.linewidth': 1.2,       
    'pdf.fonttype': 42,          
    'ps.fonttype': 42
})

import joblib
@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar."""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def plot_predicted_curve(
    inv_result: Any, 
    period: np.ndarray | list[float], 
    mode: int, 
    wave: Literal["rayleigh", "love"], 
    curve_type: Literal["phase", "group", "ellipticity"], 
    show: Literal["best", "percentage"] = "best", 
    stride: int = 1, 
    percent: float | int = 10, 
    plot_args: dict[str, Any] | None = None, 
    ax: Axes | None = None
) -> None:
    """
    Plots the modeled dispersion curves from an inversion result.

    :param inv_result: The inversion result object containing models and misfits.
    :type inv_result: object
    :param period: Array of periods (or frequencies) used for modeling.
    :type period: array_like
    :param mode: Mode number (0 for fundamental, 1 for first overtone, etc.).
    :type mode: int
    :param wave: Wave type ('rayleigh' or 'love').
    :type wave: str
    :param curve_type: Type of curve to plot ('phase', 'group', or 'ellipticity').
    :type curve_type: str
    :param show: What models to show: 'best' (single curve) or 'percentage' (ensemble). Default is 'best'.
    :type show: str, optional
    :param stride: Step size for downsampling the plotted models when show='percentage'. Default is 1.
    :type stride: int, optional
    :param percent: Top percentage of models to plot when show='percentage'. Default is 10.
    :type percent: float, optional
    :param plot_args: Dictionary of keyword arguments passed to matplotlib. Default is None.
    :type plot_args: dict, optional
    :param ax: Axes to plot on. If None, uses the current active axes. Default is None.
    :type ax: matplotlib.axes.Axes, optional
    :raises ValueError: If an invalid `curve_type` is provided.
    """
    valid_types = {"phase", "group", "ellipticity"}
    if curve_type not in valid_types:
        raise ValueError(f"Invalid curve_type: '{curve_type}'. Must be one of {valid_types}.")
    
    # Default physics parameters
    n_jobs = -1
    dc = 0.001
    dt = 0.01
    itype = {"phase": 0, "group": 1, "ellipticity": 2}
    units = {"frequency": "Hz", "period": "s"}
    
    # Model dispersion curves closure
    def get_y(thickness, velocity_p, velocity_s, density):
        c = surf96(period, thickness, velocity_p, velocity_s, density, mode, 
                   itype[curve_type], ifunc["dunkin"][wave], dc, dt)
        idx = c > 0.0
        return c[idx]
    
    # Process plot arguments securely
    _plot_args = {"type": "line", "xaxis": "period", "yaxis": "velocity", "cmap": "Oranges_r"}
    if plot_args:
        _plot_args.update(plot_args)

    plot_type = _plot_args.pop("type")
    xaxis = _plot_args.pop("xaxis")
    yaxis = _plot_args.pop("yaxis")
    cmap_name = _plot_args.pop("cmap")
    
    plot_type = "plot" if plot_type == "line" else plot_type
    
    # Set up axes
    if ax is None:
        ax = plt.gca()
    plot_func = getattr(ax, plot_type)
    
    x = 1.0 / period if xaxis == "frequency" else period
    
    if show == 'percentage':
        # Select top percentage models based on misfits
        idx = np.argsort(inv_result.misfits)
        models = inv_result.models[idx]
        misfits = inv_result.misfits[idx]
        
        n_select = int(np.floor((percent / 100) * len(idx)))
        models = models[n_select::-stride]
        misfits = misfits[n_select::-stride]
        print(f"Plotting curves from {len(misfits)} models.")

        # Make colormap mapping misfit to color
        norm = Normalize(vmin=misfits.min(), vmax=misfits.max())
        smap = ScalarMappable(norm=norm, cmap=cmap_name)
        smap.set_array([])

        with tqdm_joblib(tqdm(desc="Calculating Curves", total=len(models))):
            curves = Parallel(n_jobs=n_jobs)(delayed(get_y)(*model.T) for model in models)

        for curve, misfit in zip(curves, misfits):
            y = (1.0 / curve if yaxis == "slowness" else curve * 1e3)
            plot_func(x[: len(y)], y, color=smap.to_rgba(misfit), **_plot_args)

    elif show == "best":
        curve = get_y(*inv_result.model.T)
        y = (1.0 / curve if yaxis == "slowness" else curve * 1e3)
        plot_func(x[: len(y)], y, **_plot_args)
        
    # Customize axes labels and formatting
    ax.set_xlabel(f"{xaxis.capitalize()} [{units[xaxis]}]")
    ax.set_ylabel(f"{curve_type.capitalize()} {yaxis} [m/s]")
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(ScalarFormatter())

def plot_model(
    inv_result: Any, 
    parameter: str, 
    show: Literal["best", "mean", "percentage"] = "best", 
    stride: int = 1, 
    percent: float | int = 10, 
    zmax: float | None = None, 
    plot_args: dict[str, Any] | None = None, 
    ax: Axes | None = None, 
    cmap_on: bool = False, 
    cmap_args: dict[str, Any] | None = None, 
    cmap_range: tuple[float, float] | None = None
) -> None:
    """
    Plots the 1D depth profile of inverted parameters.

    :param inv_result: The inversion result object containing models and misfits.
    :type inv_result: object
    :param parameter: The parameter to plot ('velocity_s', 'velocity_p', 'density', 'vs', 'vp', 'rho').
    :type parameter: str
    :param show: What models to show: 'best', 'mean', or 'percentage'. Default is 'best'.
    :type show: str, optional
    :param stride: Step size for downsampling plotted models when show='percentage'. Default is 1.
    :type stride: int, optional
    :param percent: Top percentage of models to evaluate/plot. Default is 10.
    :type percent: float, optional
    :param zmax: Maximum depth for the plot. Default is None.
    :type zmax: float, optional
    :param plot_args: Keyword arguments passed to the underlying depthplot. Default is None.
    :type plot_args: dict, optional
    :param ax: Axes to plot on. Default is None.
    :type ax: matplotlib.axes.Axes, optional
    :param cmap_on: Whether to draw a colorbar based on misfit (used when show='percentage'). Default is False.
    :type cmap_on: bool, optional
    :param cmap_args: Additional arguments for the colorbar. Default is None.
    :type cmap_args: dict, optional
    :param cmap_range: Manual limits for the colormap (vmin, vmax). Default is None.
    :type cmap_range: tuple, optional
    :raises ValueError: If an invalid `parameter` is provided.
    """
    parameters = {
        "velocity_p": 1, "vp": 1,
        "velocity_s": 2, "vs": 2,
        "density": 3, "rho": 3,
    }
    
    if parameter not in parameters:
        raise ValueError(f"Invalid parameter: '{parameter}'. Choose from {list(parameters.keys())}.")
    
    i_param = parameters[parameter]
        
    _plot_args = {"cmap": "gist_ncar", "color": "black", "linewidth": 2}
    if plot_args:
        _plot_args.update(plot_args)
    cmap_name = _plot_args.pop("cmap")

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    
    if show == 'percentage':
        idx = np.argsort(inv_result.misfits)
        models = inv_result.models[idx]
        misfits = inv_result.misfits[idx]
        
        n_select = int(np.floor((percent / 100) * len(idx)))
        models = models[n_select::-stride]
        misfits = misfits[n_select::-stride]
        print(f"Plotting depth profiles for {len(misfits)} models.")

        norm_min, norm_max = cmap_range if cmap_range else (misfits.min(), misfits.max())
        norm = Normalize(vmin=norm_min, vmax=norm_max)
        smap = ScalarMappable(norm=norm, cmap=cmap_name)
        smap.set_array([])

        for model, misfit in zip(models, misfits):
            tmp_args = _plot_args.copy()
            tmp_args["color"] = smap.to_rgba(misfit)
            depthplot(model[:, 0]*1e3, model[:, i_param]*1e3, zmax, plot_args=tmp_args, ax=ax)

    elif show == "best":
        model = inv_result.model
        depthplot(model[:, 0]*1e3, model[:, i_param]*1e3, zmax, plot_args=_plot_args, ax=ax)
        
    elif show == "mean":
        idx = np.argsort(inv_result.misfits)
        models = inv_result.models[idx]
        misfits = inv_result.misfits[idx]
        
        n_select = int(np.floor((percent / 100) * len(idx)))
        models = models[:n_select+1]

        print(f"Plotting mean of {len(models)} models.")
        print(f"Misfit range: {misfits[0]:.4f} to {misfits[n_select]:.4f}")
        
        model_mean = np.squeeze(np.mean(models, axis=0))
        depthplot(model_mean[:, 0]*1e3, model_mean[:, i_param]*1e3, zmax, plot_args=_plot_args, ax=ax)
                
    labels = {
        "velocity_p": "P-wave velocity [m/s]", "vp": "$V_p$ [m/s]",
        "velocity_s": "S-wave velocity [m/s]", "vs": "$V_s$ [m/s]",
        "density": "Density [$kg/m^3$]", "rho": "$\\rho$ [$kg/m^3$]",
    }
    ax.set_xlabel(labels[parameter])
    ax.set_ylabel("Depth [m]")
    
    # Optional Colorbar Handling
    if cmap_on and show == 'percentage':
        _cmap_args = {"orientation": "vertical", "label": "Log Misfit", "location": "right"}
        if cmap_args:
            _cmap_args.update(cmap_args)
        plt.colorbar(smap, ax=ax, **_cmap_args)


def plot_model_range(
    model: Any, 
    plot_args: dict[str, Any] | None = None, 
    ax: Axes | None = None
) -> None:
    """
    Plots the prior search bounds of the inversion parameter space.

    :param model: The earth model object containing layer bounds.
    :type model: object
    :param plot_args: Keyword arguments passed to the underlying depthplot. Default is None.
    :type plot_args: dict, optional
    :param ax: Axes to plot on. Default is None.
    :type ax: matplotlib.axes.Axes, optional
    """
    d1, vs1, d2, vs2 = [], [], [], []
    
    for layer in model.layers:
        d1.append(layer.thickness[1])
        vs1.append(layer.velocity_s[0])
        d2.append(layer.thickness[0])
        vs2.append(layer.velocity_s[1])
        
    d1, vs1, d2, vs2 = map(np.array, (d1, vs1, d2, vs2))
    
    _plot_args = {"color": "black", "linewidth": 2}
    if plot_args:
        _plot_args.update(plot_args)
    
    # Calculate cumulative depths for the lower bound array
    d2[-1] = np.sum(d1) - np.sum(d2[:-1])

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 8))
    
    depthplot(d1*1e3, vs1*1e3, None, plot_args=_plot_args, ax=ax)
    depthplot(d2*1e3, vs2*1e3, None, plot_args=_plot_args, ax=ax)
    

def model_param_range(
    inv_result: Any, 
    percent: float | int = 30, 
    stride: int | None = None, 
    plot_nu: bool = False
) -> None:
    """
    Calculates and prints statistical percentiles (0, 25, 50, 75, 100) 
    for the top performing models in an inversion ensemble.

    :param inv_result: The inversion result object containing the ensemble data.
    :type inv_result: object
    :param percent: Top percentage of models to evaluate. Default is 30.
    :type percent: float, optional
    :param stride: Step size for downsampling evaluated models. Default is None.
    :type stride: int, optional
    :param plot_nu: Whether to plot histograms of Poisson's Ratio distributions. Default is False.
    :type plot_nu: bool, optional
    """
    idx = np.argsort(inv_result.misfits)
    models = inv_result.xs[idx]

    n_select = int(np.floor((percent / 100) * len(idx)))
    if stride is None:
        stride = max(1, n_select // 4)
        
    models = models[n_select::-stride]
    print(f"Evaluating {len(models)} models from the ensemble.")
    
    n_layer = int((len(models[0]) + 1) // 3)
    
    # Calculate percentiles
    h = np.percentile(models[:, :n_layer-1], [0, 25, 50, 75, 100], axis=0)
    vs = np.percentile(models[:, n_layer-1:2*n_layer-1], [0, 25, 50, 75, 100], axis=0)
    nu = np.percentile(models[:, 2*n_layer-1:], [0, 25, 50, 75, 100], axis=0)
    
    for i in range(n_layer):
        print(f"--- Layer {i} ---")
        if i < n_layer - 1:
            print(f"Thickness [km] : {h[0,i]:.4f}, {h[1,i]:.4f}, {h[2,i]:.4f}, {h[3,i]:.4f}, {h[4,i]:.4f}")
        print(f"Vs [km/s]      : {vs[0,i]:.4f}, {vs[1,i]:.4f}, {vs[2,i]:.4f}, {vs[3,i]:.4f}, {vs[4,i]:.4f}")
        print(f"Poisson (Nu)   : {nu[0,i]:.4f}, {nu[1,i]:.4f}, {nu[2,i]:.4f}, {nu[3,i]:.4f}, {nu[4,i]:.4f}\n")

    if plot_nu:
        nu_data = models[n_select::-1, 2*n_layer-1:]
        for i in range(n_layer):
            plt.figure(figsize=(6, 4))
            plt.hist(nu_data[:, i], bins=20, edgecolor='black', alpha=0.7)
            plt.title(f"Poisson's Ratio Distribution - Layer {i}")
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.show()
        

def get_mean_model(
    inv_result: Any, 
    percent: float | int = 30
) -> np.ndarray:
    """
    Extracts the mean model from the top percentage of inversion results.

    :param inv_result: The inversion result object containing models and misfits.
    :type inv_result: object
    :param percent: Top percentage of models to average. Default is 30.
    :type percent: float, optional
    :returns: The averaged 1D model array.
    :rtype: np.ndarray
    """
    idx = np.argsort(inv_result.misfits)
    models = inv_result.models[idx]
    misfits = inv_result.misfits[idx]

    n_select = int(np.floor((percent / 100) * len(idx)))
    models = models[:n_select+1]
    
    print(f"Extracted mean of top {len(models)} models.")
    print(f"Misfit range: {misfits[0]:.4f} to {misfits[n_select]:.4f}")

    return np.squeeze(np.mean(models, axis=0))

def check(
    all_results: list[Any], 
    positions: np.ndarray | list[float | int], 
    index: int
) -> None:
    """
    Instantly prints the stats and plots the misfit for any station index.

    :param all_results: List of all inversion results.
    :type all_results: list
    :param positions: List or array of station positions/distances.
    :type positions: array_like
    :param index: Index of the station to check.
    :type index: int
    """
    res = all_results[index]
    dist = positions[index]
    
    print(f"=== Station at {dist}m (Index {index}) ===")
    print(res)
    res.plot_misfit()
    plt.title(f"Misfit for {dist}m")
    plt.show()
    
def save_profile_plot(
    res: Any, 
    prior_model: Any, 
    dist: float | int | str, 
    save_dir: str = "../results/inv_profiles", 
    xlim: list[float | int] | tuple[float | int, float | int] = [0, 1100], 
    zmax: float | int = 150, 
    percent: float | int = 10
) -> None:
    """
    Generates and saves the 1D Vs profile plot to disk without displaying it inline.

    :param res: The inversion result object to plot.
    :type res: object
    :param prior_model: The prior search bounds model object.
    :type prior_model: object
    :param dist: Station distance or identifier for the filename and title.
    :type dist: float | int | str
    :param save_dir: Directory to save the resulting PNG image. Default is "../results/inv_profiles".
    :type save_dir: str, optional
    :param xlim: X-axis limits (Vs range). Default is [0, 1100].
    :type xlim: list, optional
    :param zmax: Maximum depth to display on the Y-axis. Default is 150.
    :type zmax: float, optional
    :param percent: Top percentage of ensemble models to overlay. Default is 10.
    :type percent: float, optional
    """
    os.makedirs(save_dir, exist_ok=True)
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(9, 10))
    ax.grid(True, which='major', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.grid(True, which='minor', linestyle=':', linewidth=0.5, alpha=0.4)
    ax.minorticks_on()

    # A. Prior Bounds (Black Solid)
    plot_model_range(prior_model, plot_args={"color": "black", "linestyle": "-", "linewidth": 2}, ax=ax)
    
    # B. All Models (Ensemble Density - Grayscale)
    plot_model(res, parameter="vs", show="percentage", percent=100, stride=10, zmax=zmax, 
               cmap_on=True, cmap_range=(0.8, 2.0), # Matches your Hayashi reference
               plot_args={"cmap": "Greys_r", "alpha": 0.05, "linewidth": 1}, ax=ax)
    
    # C. Top 30% Mean (Cyan Dashed)
    plot_model(res, parameter="vs", show="mean", percent=30, zmax=zmax, 
               plot_args={"color": "cyan", "linewidth": 2.5, "linestyle": "--"}, ax=ax)
    
    # D. Best Model (Red Solid)
    plot_model(res, parameter="vs", show="best", zmax=zmax, 
               plot_args={"color": "red", "linewidth": 3.5, "linestyle": "-"}, ax=ax)

    ax.set_title(f"Inverted 1D $V_s$ Profile | Station: {dist}m", pad=15)
    ax.set_xlabel("$V_s$ (m/s)")
    ax.set_ylabel("Depth (m)")
    ax.set_xlim(xlim)

    legend_elements = [
        Line2D([0], [0], color='red', lw=3.5, label='Best Inverted Model'),
        Line2D([0], [0], color='cyan', lw=2.5, linestyle='--', label='Mean of Top 30% Models'),
        Line2D([0], [0], color='gray', lw=2, alpha=0.5, label='Evaluated Models (Misfit Scale)'),
        Line2D([0], [0], color='black', lw=2, label='Prior Search Bounds')
    ]

    ax.legend(handles=legend_elements, loc='lower left', framealpha=0.95, edgecolor='black')
    
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"Vs_profile_{dist}m.png"), dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_dispersion_fit(
    all_results: list[Any], 
    positions: np.ndarray | list[float | int], 
    index: int, 
    obs_dir: str = "../results/inv_inputs",
    percent: float | int = 10, 
    figsize: tuple[float | int, float | int] = (9, 5), 
    title: str | None = None, 
    save_path: str | None = None
) -> None:
    """
    Loads observed DAS dispersion data for a specific station index and plots it 
    against the theoretical dispersion curves (ensemble and best fit) from the inversion.

    :param all_results: List of all inversion result objects.
    :type all_results: list[Any]
    :param positions: Array or list of spatial distances along the cable.
    :type positions: array_like
    :param index: The integer index of the station/result to plot.
    :type index: int
    :param obs_dir: Directory containing the observed dispersion text files. Default is "../results/inv_inputs".
    :type obs_dir: str, optional
    :param percent: Top percentage of the ensemble models to plot. Default is 10.
    :type percent: float | int, optional
    :param figsize: Dimensions of the generated figure (width, height). Default is (9, 5).
    :type figsize: tuple, optional
    :param title: The title to display above the plot. If None, defaults to "Dispersion Fit | Station: {dist}m".
    :type title: str | None, optional
    :param save_path: If provided, saves the figure to this file path instead of displaying it inline. Default is None.
    :type save_path: str | None, optional
    :raises FileNotFoundError: If the observation text file for the specified station is missing.
    """
    # 1. Extract the specific inversion result and distance
    res = all_results[index]
    dist = positions[index]

    # 2. Load the observed data
    obs_file = os.path.join(obs_dir, f"dispersion_{int(dist):04d}m.txt")
    if not os.path.exists(obs_file):
        raise FileNotFoundError(f"Could not find observed data file: {obs_file}")
        
    obs_data = np.loadtxt(obs_file)
    freq_obs = obs_data[:, 0]
    vel_obs = obs_data[:, 1]
    
    # 3. Calculate period for the theoretical modeling
    disp_period = 1.0 / freq_obs

    # 4. Set up the plot
    fig, ax = plt.subplots(figsize=figsize)

    # A. Plot the ensemble of theoretical dispersion curves
    plot_predicted_curve(
        res, 
        disp_period, 
        mode=0, 
        wave='rayleigh', 
        curve_type='phase', 
        show="percentage", 
        percent=percent, 
        plot_args={"xaxis": "frequency", "cmap": "turbo", "alpha": 0.1}, 
        ax=ax
    )

    # B. Plot the single best theoretical curve
    plot_predicted_curve(
        res, 
        disp_period, 
        mode=0, 
        wave='rayleigh', 
        curve_type='phase', 
        show="best", 
        plot_args={"xaxis": "frequency", "color": "black", "linewidth": 2, "linestyle": "--"}, 
        ax=ax
    )

    # C. Overlay the actual DAS observations
    ax.plot(freq_obs, vel_obs, 'ro', label='Observed Data (DAS)', markersize=6, alpha=0.9)

    # Formatting
    ax.grid(True, linestyle=':', alpha=0.6)
    
    handles, labels = ax.get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    if by_label:
        ax.legend(by_label.values(), by_label.keys(), fontsize=12)

    if title is None:
        title = f"Dispersion Fit | Station: {dist}m"
    ax.set_title(title, fontsize=14)
    
    plt.tight_layout()
    
    # Save if a path is provided, otherwise just display it inline
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure successfully saved to: {save_path}")
        plt.close(fig)
    else:
        plt.show()

def animate_saved_profiles(
    positions: np.ndarray | list[float | int], 
    img_dir: str = "../results/inv_profiles", 
    prefix: str = "Vs_profile_", 
    suffix: str = "m.png", 
    interval: int = 200, 
    figsize: tuple[float | int, float | int] = (8, 9)
) -> None:
    """
    Stitches saved PNG images into an interactive Jupyter video player.

    :param positions: Array or list of positions used to match file names.
    :type positions: array_like
    :param img_dir: Directory where the images are stored. Default is "../results/inv_profiles".
    :type img_dir: str, optional
    :param prefix: File name prefix before the position value. Default is "Vs_profile_".
    :type prefix: str, optional
    :param suffix: File name suffix after the position value. Default is "m.png".
    :type suffix: str, optional
    :param interval: Delay between frames in milliseconds. Default is 200.
    :type interval: int, optional
    :param figsize: Tuple specifying figure dimensions. Default is (8, 9).
    :type figsize: tuple, optional
    """
    print("Stitching saved images into an animation...")

    # 1. Set up a blank figure
    fig, ax = plt.subplots(figsize=figsize)
    ax.axis('off') # Turn off the axes since the PNG already has its own axes drawn!
    fig.tight_layout()

    # 2. Load the very first image to initialize the plot
    first_dist = positions[0]
    first_img_path = os.path.join(img_dir, f"{prefix}{first_dist}{suffix}")
    
    if not os.path.exists(first_img_path):
        print(f"Error: Could not find the first image at {first_img_path}. Check your directory!")
        plt.close(fig)
        return

    first_img = mpimg.imread(first_img_path)

    # Display the first image
    im = ax.imshow(first_img)

    # 3. Define the update function for FuncAnimation
    def update(frame_idx):
        """Reads the next image from disk and updates the display."""
        dist = positions[frame_idx]
        img_path = os.path.join(img_dir, f"{prefix}{dist}{suffix}")
        
        # If for some reason an image is missing, skip it gracefully
        if os.path.exists(img_path):
            img = mpimg.imread(img_path)
            im.set_array(img)
            
        return [im]

    # 4. Create the animation
    anim = FuncAnimation(
        fig, 
        update, 
        frames=len(positions), 
        interval=interval, 
        blit=True
    )

    # 5. Prevent Matplotlib from showing a duplicate static plot
    plt.close(fig)

    # 6. Display the interactive video player in Jupyter
    print("Loading video player...")
    display(HTML(anim.to_jshtml()))

def plot_2d_contour_section(
    positions: np.ndarray | list[float | int], 
    z_grid: np.ndarray | list[float | int], 
    vs_2d_matrix: np.ndarray, 
    max_depth: float | int = 120, 
    vmin: float | int = 200, 
    vmax: float | int = 600, 
    levels: int = 50, 
    cmap: str = 'turbo', 
    figsize: tuple[float | int, float | int] = (12, 5), 
    smooth_sigma: tuple[float, float] = (1, 2), 
    tick_step: int = 100, 
    contour: bool = False,
    x_flip: bool = False,
    x_interp_step: float | None = None,
    smooth_units: str = 'index',
    max_resolved_depth: float | int | None = None,
    save_path: str | None = None
) -> None:
    """
    Plots a 2D contoured shear-wave velocity cross-section.

    :param positions: Array of horizontal coordinates (e.g., Distance Along Cable).
    :type positions: array_like
    :param z_grid: Array of depth coordinates.
    :type z_grid: array_like
    :param vs_2d_matrix: 2D array of shear-wave velocity values mapping to (z_grid, positions).
    :type vs_2d_matrix: np.ndarray
    :param max_depth: Maximum depth to display on the Y-axis. Default is 120.
    :type max_depth: float | int, optional
    :param vmin: Minimum velocity for the colormap. Default is 200.
    :type vmin: float | int, optional
    :param vmax: Maximum velocity for the colormap. Default is 600.
    :type vmax: float | int, optional
    :param levels: Number of contour levels to plot. Default is 50.
    :type levels: int, optional
    :param cmap: Matplotlib colormap to use. Default is 'turbo'.
    :type cmap: str, optional
    :param figsize: Dimensions of the figure. Default is (12, 5).
    :type figsize: tuple, optional
    :param smooth_sigma: Standard deviation for Gaussian smoothing. Default is (1, 2).
    :type smooth_sigma: tuple, optional
    :param tick_step: Step size for the colorbar ticks. Default is 100.
    :type tick_step: int, optional
    :param contour: Whether to overlay discrete contour lines. Default is False.
    :type contour: bool, optional
    :param x_flip: Whether to invert the X-axis (e.g., to match map orientation). Default is False.
    :type x_flip: bool, optional
    :param x_interp_step: If set, linearly interpolate the columns onto a uniform
        horizontal grid with this spacing (same units as ``positions``, e.g. metres)
        BEFORE contouring. With sparse virtual shots this removes the vertical
        banding caused by contouring straight between far-apart columns. Default None
        (plot on the raw shot positions, legacy behaviour).
    :type x_interp_step: float, optional
    :param smooth_units: ``'index'`` (legacy) treats ``smooth_sigma`` as samples;
        ``'physical'`` treats it as ``(sigma_z, sigma_x)`` in physical units (metres)
        and converts to samples via the grid spacing, so the blur is independent of
        grid resolution. Default ``'index'``.
    :type smooth_units: str, optional
    :param max_resolved_depth: If set, draw a dashed line + label at this depth to
        mark the maximum reliably resolved depth (below it the model is poorly
        constrained, e.g. the half-space). Default None.
    :type max_resolved_depth: float | int, optional
    :param save_path: If provided, saves the figure to this path instead of showing inline. Default is None.
    :type save_path: str, optional
    """
    fig, ax = plt.subplots(figsize=figsize)

    positions = np.asarray(positions, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    M = np.asarray(vs_2d_matrix, dtype=float)

    # Accept either orientation: the matrix must be (len(z_grid), len(positions)).
    # If it arrives transposed as (len(positions), len(z_grid)) -- e.g. the raw
    # list-of-profiles before the .T in the inversion loop -- orient it here so
    # the call works regardless of run order.
    n_z, n_x = z_grid.size, positions.size
    if M.shape == (n_x, n_z) and n_x != n_z:
        M = M.T
    if M.shape != (n_z, n_x):
        raise ValueError(
            f"vs_2d_matrix has shape {M.shape}; expected (len(z_grid), len(positions)) "
            f"= ({n_z}, {n_x}) or its transpose ({n_x}, {n_z})."
        )

    # contourf and lateral interpolation need columns sorted by position.
    order = np.argsort(positions)
    shot_positions = positions[order]
    M = M[:, order]

    # Optional lateral densification onto a uniform grid BEFORE contouring.
    # With sparse virtual shots, contouring straight between far-apart columns
    # produces vertical banding; interpolating to a fine grid removes it.
    if x_interp_step:
        x_plot = np.arange(shot_positions.min(),
                           shot_positions.max() + x_interp_step, x_interp_step)
        M = np.vstack([np.interp(x_plot, shot_positions, M[r]) for r in range(M.shape[0])])
    else:
        x_plot = shot_positions

    # Smoothing. 'physical' treats smooth_sigma as (sigma_z, sigma_x) in METRES
    # and converts to samples via the grid spacing, so the blur is independent of
    # grid resolution. 'index' (legacy) treats smooth_sigma as samples.
    if smooth_sigma is not None and tuple(smooth_sigma) != (0, 0):
        if str(smooth_units).lower() == 'physical':
            dz = float(np.median(np.diff(z_grid))) if z_grid.size > 1 else 1.0
            dx = float(np.median(np.diff(x_plot))) if x_plot.size > 1 else 1.0
            sigma_eff = (smooth_sigma[0] / dz, smooth_sigma[1] / dx)
        else:
            sigma_eff = smooth_sigma
        plot_matrix = gaussian_filter(M, sigma=sigma_eff)
    else:
        plot_matrix = M

    X, Z = np.meshgrid(x_plot, z_grid)

    # Define levels for both the colorbar and the lines
    contour_levels = np.linspace(vmin, vmax, levels)
    tick_levels = np.arange(vmin, vmax + tick_step, tick_step)

    # 1. Filled Contours
    cf = ax.contourf(X, Z, plot_matrix, levels=contour_levels, cmap=cmap, extend='both')

    # 2. Contour Lines (conditional)
    if contour:
        cl = ax.contour(X, Z, plot_matrix, levels=tick_levels, colors='black', linewidths=0.5, alpha=0.3)
        # Label the lines
        ax.clabel(cl, inline=True, fontsize=8, fmt='%1.0f')

    # Colorbar with fixed ticks
    cbar = fig.colorbar(cf, ax=ax, pad=0.02, ticks=tick_levels)
    cbar.set_label('$V_s$ (m/s)', fontsize=12)

    # Formatting
    ax.set_ylim(max_depth, 0)
    
    # Set explicit X-limits and handle the flip
    x_min, x_max = np.min(shot_positions), np.max(shot_positions)
    if x_flip:
        ax.set_xlim(x_max, x_min)
    else:
        ax.set_xlim(x_min, x_max)

    # Mark the maximum reliably resolved depth (below it the model is poorly
    # constrained, e.g. the half-space).
    if max_resolved_depth is not None:
        ax.axhline(max_resolved_depth, color='white', ls='--', lw=1.2, alpha=0.85)
        ax.text(0.015, max_resolved_depth,
                f' max resolved ≈ {max_resolved_depth:.0f} m',
                transform=ax.get_yaxis_transform(), va='bottom', ha='left',
                color='white', fontsize=8)

    ax.set_title(f"Contoured 2D Shear-Wave Velocity ($V_s$) Profile", fontsize=16, pad=15)
    ax.set_xlabel("Distance Along Cable (m)", fontsize=14)
    ax.set_ylabel("Depth (m)", fontsize=14)
    
    # Show virtual shot locations
    ax.scatter(shot_positions, np.zeros_like(shot_positions), marker='v', color='black',
               s=50, clip_on=False, label='Virtual Shots', zorder=5)
    ax.legend(loc='upper left', fontsize=12)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Figure successfully saved to: {save_path}")
        
    plt.show()

def animate_sensitivity_kernels(
    positions: np.ndarray, 
    vs_matrix: np.ndarray, 
    z_grid: np.ndarray, 
    test_frequencies: list[float] | None = None, 
    vp_vs_ratio: float = 2.0, 
    density: float = 1.0, 
    x_max: float = 0.08,
    ylim: Tuple[float, float] | None = None,
    max_model_depth: float | None = None,
    normalize: bool = False,
    step: int | None = None,
    interval_ms: int = 150,
    save_indices: List[int] | None = None,
    save_dir: str = "../results/inv_sensitivity_urban",
    save_fmt: str = "png",
    save_dpi: int = 300,
) -> FuncAnimation:
    """
    Generates Rayleigh wave sensitivity kernels along a 2D seismic profile.
    
    :param positions: 1D array of horizontal positions along the cable (in meters).
    :param vs_matrix: 2D shear wave velocity matrix (depth x positions) in m/s.
    :param z_grid: 1D array of depths (in meters).
    :param test_frequencies: Frequencies to test in Hz. Default is [2.0, 3.0, 4.0, 5.0, 6.0].
    :param vp_vs_ratio: Ratio used to estimate Vp from Vs. Default is 2.0.
    :param density: Constant density assumption in g/cm^3. Default is 1.0.
    :param x_max: Maximum limit for the X-axis to keep the animation stable. Default is 0.08.
    :param ylim: Optional tuple (min_depth, max_depth) to manually set the y-axis limits.
    :param step: Frame step size. E.g., step=5 animates every 5th position. Default is None.
    :param interval_ms: Milliseconds between frames. Lower is faster. Default is 150.
    :param max_model_depth: If set, extend the 1D model below the inverted grid
        (holding the deepest/half-space Vs constant) down to this depth before
        computing kernels.
    :param normalize: If True, divide each kernel by its layer thickness
        (sensitivity per metre) so the thick terminal half-space is comparable to
        the thin layers above it. Default False.
    :param save_indices: Optional list of frame indices to save as static high-res images.
    :param save_dir: Directory where the static frames will be saved. Default is "../results/inv_sensitivity_urban".
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    :returns: The constructed Matplotlib FuncAnimation object.
    """
    if test_frequencies is None:
        test_frequencies = [2.0, 3.0, 4.0, 5.0, 6.0]
    
    # 1. Pre-calculate static variables.
    z_depths = np.abs(np.asarray(z_grid, dtype=float))
    if max_model_depth is not None and max_model_depth > z_depths.max():
        dz = float(np.median(np.diff(z_depths)))
        z_ext = np.arange(z_depths.max() + dz, float(max_model_depth) + dz, dz)
        z_model = np.concatenate([z_depths, z_ext])
    else:
        z_model = z_depths
    n_ext = z_model.size - z_depths.size
    thickness_m = np.append(np.diff(z_model), 10.0)  # last entry = half-space
    thickness_km = thickness_m / 1000.0

    # 2. Setup the static figure
    fig, ax = plt.subplots(figsize=(5, 7), layout="constrained")
    lines = []
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(test_frequencies)))

    for idx, f in enumerate(test_frequencies):
        line, = ax.plot([], [], label=f"{f} Hz", linewidth=2.5, color=colors[idx])
        lines.append(line)

    if ylim is not None:
        ax.set_ylim(max(ylim), min(ylim))  # Force inverted y-axis
    else:
        ax.set_ylim(np.max(z_model), 0)
        
    ax.set_xlim(0, x_max)
    
    ax.set_xlabel("Sensitivity Kernel ($\partial c / \partial V_s$)", fontsize=14)
    ax.set_ylabel("Depth (m)", fontsize=14)
    
    title = ax.set_title("", fontsize=16, pad=20)

    ax.grid(True, linestyle='--', alpha=0.7)
    ax.legend(loc="lower right", fontsize=12)
    
    saved_frames = set()

    # Animation internal functions
    def init():
        for line in lines:
            line.set_data([], [])
        title.set_text("")
        return lines + [title]

    def update(idx):
        pos_m = positions[idx]
        vs_1d_ms = np.asarray(vs_matrix[:, idx], dtype=float)

        if n_ext > 0:
            vs_1d_ms = np.concatenate([vs_1d_ms, np.full(n_ext, vs_1d_ms[-1])])

        vs_kms = vs_1d_ms / 1000.0
        vp_kms = vs_kms * vp_vs_ratio
        rho_gcm3 = np.full_like(vs_kms, density)
        velocity_model = np.column_stack((thickness_km, vp_kms, vs_kms, rho_gcm3))

        ps = PhaseSensitivity(*velocity_model.T)

        for f_idx, f in enumerate(test_frequencies):
            period = 1.0 / f
            k = ps(period, mode=0, wave="rayleigh", parameter="velocity_s")
            kernel = np.asarray(k.kernel, dtype=float)
            if normalize:
                kernel = kernel / np.asarray(thickness_m, dtype=float)
            lines[f_idx].set_data(kernel, k.depth * 1000.0)

        title.set_text(f"Rayleigh Wave Sensitivity\nat Position = {pos_m:.1f} m")

        if save_indices is not None and idx in save_indices:
            if idx not in saved_frames:
                os.makedirs(save_dir, exist_ok=True)
                save_path = os.path.join(save_dir, f"Sensitivity_Kernel_Pos_{pos_m:.1f}m.{save_fmt}")
                fig.savefig(save_path, dpi=save_dpi, bbox_inches="tight", facecolor="white")
                saved_frames.add(idx)

        return lines + [title]

    if step is None:
        step = max(1, len(positions) // 100) 
        
    frames_to_render = range(0, len(positions), step)

    ani = FuncAnimation(
        fig, update, frames=frames_to_render, 
        init_func=init, blit=True, interval=interval_ms
    )

    plt.close(fig) 
    return HTML(ani.to_jshtml()) 

def save_sensitivity_kernel_plots(
    positions: np.ndarray, 
    vs_matrix: np.ndarray, 
    z_grid: np.ndarray, 
    test_frequencies: list[float] | None = None, 
    vp_vs_ratio: float = 2.0, 
    density: float = 1.0, 
    x_max: float = 0.08,
    ylim: Tuple[float, float] | None = None,
    max_model_depth: float | None = None,
    normalize: bool = False,
    step: int | None = None,
    save_dir: str = "../results/inv_sensitivity_urban",
    save_fmt: str = "png",
    save_dpi: int = 300,
    figsize: Tuple[float, float] = (5, 7)
) -> None:
    """
    Batch processes and saves static Rayleigh wave sensitivity kernels.
    
    This function avoids the overhead of animation rendering by directly saving 
    a PNG plot for each specified horizontal position along the grid.
    
    :param step: Step size across the positions array. If step=10, every 10th position is saved.
                 If None, defaults to step=1 (saves every single position).
    :param ylim: Optional tuple (min_depth, max_depth) to manually set the y-axis limits.
    :param save_dir: Directory where the static frames will be saved. Default is "../results/inv_sensitivity_urban".
    :param save_fmt: Image format for the saved frames (e.g., "png"). Default is "png".
    :param save_dpi: Resolution for the saved frames. Default is 300.
    """
    os.makedirs(save_dir, exist_ok=True)
    if test_frequencies is None:
        test_frequencies = [2.0, 3.0, 4.0, 5.0, 6.0]
        
    z_depths = np.abs(np.asarray(z_grid, dtype=float))
    if max_model_depth is not None and max_model_depth > z_depths.max():
        dz = float(np.median(np.diff(z_depths)))
        z_ext = np.arange(z_depths.max() + dz, float(max_model_depth) + dz, dz)
        z_model = np.concatenate([z_depths, z_ext])
    else:
        z_model = z_depths
    n_ext = z_model.size - z_depths.size
    thickness_m = np.append(np.diff(z_model), 10.0)
    thickness_km = thickness_m / 1000.0

    if step is None:
        step = 1
    indices_to_render = range(0, len(positions), step)

    colors = plt.cm.tab10(np.linspace(0, 1, len(test_frequencies)))

    for idx in tqdm(indices_to_render, desc="Saving Sensitivity Kernels"):
        fig, ax = plt.subplots(figsize=figsize, layout="constrained", dpi=save_dpi)
        
        pos_m = positions[idx]
        vs_1d_ms = np.asarray(vs_matrix[:, idx], dtype=float)

        if n_ext > 0:
            vs_1d_ms = np.concatenate([vs_1d_ms, np.full(n_ext, vs_1d_ms[-1])])

        vs_kms = vs_1d_ms / 1000.0
        vp_kms = vs_kms * vp_vs_ratio
        rho_gcm3 = np.full_like(vs_kms, density)
        velocity_model = np.column_stack((thickness_km, vp_kms, vs_kms, rho_gcm3))

        ps = PhaseSensitivity(*velocity_model.T)

        for f_idx, f in enumerate(test_frequencies):
            period = 1.0 / f
            k = ps(period, mode=0, wave="rayleigh", parameter="velocity_s")
            kernel = np.asarray(k.kernel, dtype=float)
            if normalize:
                kernel = kernel / np.asarray(thickness_m, dtype=float)
            
            ax.plot(kernel, k.depth * 1000.0, label=f"{f} Hz", linewidth=2.5, color=colors[f_idx])

        if ylim is not None:
            ax.set_ylim(max(ylim), min(ylim))
        else:
            ax.set_ylim(np.max(z_model), 0)
            
        ax.set_xlim(0, x_max)
        ax.set_xlabel("Sensitivity Kernel ($\partial c / \partial V_s$)", fontsize=14)
        ax.set_ylabel("Depth (m)", fontsize=14)
        ax.set_title(f"Rayleigh Wave Sensitivity\nat Position = {pos_m:.1f} m", fontsize=16, pad=20)
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.legend(loc="lower right", fontsize=12)

        save_path = os.path.join(save_dir, f"Sensitivity_Kernel_Pos_{pos_m:.1f}m.{save_fmt}")
        fig.savefig(save_path, bbox_inches="tight", facecolor="white")
        plt.close(fig)