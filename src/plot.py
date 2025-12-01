"""
:module: src/plot.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Plotting utilities for dispersion imaging (f–v panels)
          and dispersion curve picks.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from utils import convert_to_numpy

# 1. Plot f–v dispersion panel
def plot_fv_panel(
    fv_panel,
    f_axis,
    v_axis,
    title="Dispersion Image (f–v panel)",
    cmap="viridis",
    figsize=(7, 5),
    save_path=None,
    show=True,
):
    """
    Plot dispersion image: velocity vs frequency.

    Parameters
    ----------
    fv_panel : (nv × nf) tensor or array
    f_axis   : (nf,) frequency axis
    v_axis   : (nv,) velocity axis
    """

    P   = convert_to_numpy(fv_panel)
    f   = convert_to_numpy(f_axis)
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

# 2. Plot f–v panel + pick curve overlay
def plot_fv_with_pick(
    fv_panel,
    f_axis,
    v_axis,
    pick_curve,
    title="Dispersion with Picked Curve",
    cmap="viridis",
    figsize=(7, 5),
    save_path=None,
    show=True,
):
    """
    Plot dispersion image and overlay the dispersion pick.
    """

    P   = convert_to_numpy(fv_panel)
    f   = convert_to_numpy(f_axis)
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

    # Overlay pick curve
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

# 3. Plot dispersion pick as v(f)
def plot_pick_curve(
    f_axis,
    pick_curve,
    title="Dispersion Curve",
    figsize=(6, 4),
    save_path=None,
    show=True,
):
    """
    Plot v(f) curve only.
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