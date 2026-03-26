"""
:module: das_geo.py
:author: Benz Poobua
:email: spoobua (at) stanford.edu
:org: Stanford University
:license: MIT
:purpose: Geospatial Mapping and 2D/3D Model Interpolation Utilities.
Provides tools for plotting Web Mercator basemaps, adding map scale/decorations,
coordinate transformations, and 2D/3D velocity/data model interpolation.
Modified from Haipeng Li's code
"""

import contextily as ctx
import matplotlib.pyplot as plt
import numpy as np
from typing import Optional, Tuple, List, Union, Dict, Any

from matplotlib_scalebar.scalebar import ScaleBar
from matplotlib_map_utils.core.north_arrow import north_arrow
from obspy.geodetics import degrees2kilometers, gps2dist_azimuth, locations2degrees
from pyproj import Transformer
from rasterio.enums import Resampling
from scipy.interpolate import griddata, interpn
from scipy.ndimage import gaussian_filter

# ==========================================
# 1. Map Plotting & Decoration Utilities
# ==========================================

def add_basemap(
    ax: plt.Axes,
    lon_min: float,
    lon_max: float,
    lat_min: float,
    lat_max: float,
    zoom: Union[int, str] = "auto",
    source: Any = None,
    interpolation: str = "bilinear",
    attribution: Optional[str] = None,
    attribution_size: int = 8,
    reset_extent: bool = True,
    crs: Optional[str] = None,
    resampling: Resampling = Resampling.bilinear,
    zoom_adjust: Optional[int] = None,
    **extra_imshow_args: Any,
) -> Transformer:
    """
    Adds a basemap to a matplotlib axis using contextily, converting from lat/lon to Web Mercator.

    Args:
        ax: The matplotlib axes to plot on.
        lon_min, lon_max: Longitude boundaries.
        lat_min, lat_max: Latitude boundaries.
        zoom: Map zoom level ('auto' or integer).
        source: Contextily provider (e.g., ctx.providers.CartoDB.Voyager).
        interpolation: Image interpolation method.
        attribution: Custom attribution text (False to disable).
        attribution_size: Font size for attribution.
        reset_extent: Whether to strictly force the axis limits.
        crs: Coordinate reference system of the basemap.
        resampling: Rasterio resampling method.
        zoom_adjust: Adjustment to the auto-calculated zoom level.

    Returns:
        Transformer: A pyproj transformer object for converting EPSG:4326 to EPSG:3857.
    """
    transformer = Transformer.from_crs(crs_from="EPSG:4326", crs_to="EPSG:3857", always_xy=True)

    x_min, y_min = transformer.transform(lon_min, lat_min)
    x_max, y_max = transformer.transform(lon_max, lat_max)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    ctx.add_basemap(
        ax,
        zoom=zoom,
        source=source,
        interpolation=interpolation,
        attribution=attribution,
        attribution_size=attribution_size,
        reset_extent=reset_extent,
        crs=crs,
        resampling=resampling,
        zoom_adjust=zoom_adjust,
        **extra_imshow_args,
    )

    return transformer

def add_ticks(
    ax: plt.Axes,
    unit: str = "degree",
    interval: float = 0.1,
    rotation_x: float = 0,
    rotation_y: float = 0,
) -> None:
    """
    Adds formatted tick marks to a map axis.

    Args:
        ax: The matplotlib axes.
        unit: 'degree', 'km', or 'm'.
        interval: Spacing between ticks.
        rotation_x: Rotation angle for x-axis labels.
        rotation_y: Rotation angle for y-axis labels.
    """
    interval_str = str(interval)
    nround = len(interval_str.split(".")[1]) + 1 if ("." in interval_str and unit == "latlon") else \
             len(interval_str.split(".")[1]) if "." in interval_str else 0

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    if unit == "degree":
        transformer = Transformer.from_crs("EPSG:3857", "EPSG:4326", always_xy=True)
        # Approximate 1 degree at equator ~ 111.32 km
        x_ticks = np.linspace(xlim[0], xlim[1], int((xlim[1] - xlim[0]) / (interval * 111320)))
        y_ticks = np.linspace(ylim[0], ylim[1], int((ylim[1] - ylim[0]) / (interval * 111320)))

        x_labels = [f"{transformer.transform(tick, ylim[0])[0]:.{nround}f}" for tick in x_ticks]
        y_labels = [f"{transformer.transform(xlim[0], tick)[1]:.{nround}f}" for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_xticklabels([f"{label}°" for label in x_labels])
        ax.set_yticks(y_ticks)
        ax.set_yticklabels([f"{label}°" for label in y_labels])

    elif unit in ["km", "m"]:
        locator_interval = interval * 1000 if unit == "km" else interval
        x_ticks = plt.MultipleLocator(locator_interval).tick_values(*xlim)
        y_ticks = plt.MultipleLocator(locator_interval).tick_values(*ylim)

        divisor = 1000 if unit == "km" else 1
        xtick_labels = [f"{(tick - xlim[0]) / divisor:.{nround}f}" for tick in x_ticks]
        ytick_labels = [f"{(tick - ylim[0]) / divisor:.{nround}f}" for tick in y_ticks]

        ax.set_xticks(x_ticks)
        ax.set_yticks(y_ticks)
        ax.set_xticklabels(xtick_labels)
        ax.set_yticklabels(ytick_labels)

    plt.setp(ax.get_xticklabels(), rotation=rotation_x)
    plt.setp(ax.get_yticklabels(), rotation=rotation_y)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)

def add_scale(
    ax: plt.Axes,
    dx: float = 1.0,
    location: str = "lower right",
    length_fraction: float = 0.1,
    color: str = "black",
    box_alpha: float = 0.5,
    box_color: str = "white",
    scale_loc: str = "top",
    font_properties: Optional[Dict] = None,
) -> None:
    """
    Adds a dynamic scale bar to the map.

    Args:
        ax: The matplotlib axes.
        dx: Distance mapped to 1 unit. Default is 1.0 (for Web Mercator EPSG:3857).
        location: Matplotlib legend-style location string.
        length_fraction: Percentage of the axis width the scale bar should target.
        color: Color of the text and scale bar.
        box_alpha: Transparency of the background box.
        box_color: Color of the background box.
        scale_loc: Position of the scale text relative to the bar.
        font_properties: Dictionary of font settings.
    """
    if font_properties is None:
        font_properties = {"size": 10}

    scalebar = ScaleBar(
        dx=dx,
        units="m",
        dimension="si-length",
        length_fraction=length_fraction,
        color=color,
        box_alpha=box_alpha,
        box_color=box_color,
        scale_loc=scale_loc,
        font_properties=font_properties,
    )
    scalebar.location = location
    ax.add_artist(scalebar)

def add_north_arrow(
    ax: plt.Axes, 
    location: str = "lower left", 
    scale: float = 0.25, 
    fontsize: int = 12
) -> None:
    """
    Adds a formatted north arrow to the map using matplotlib_map_utils.
    
    Args:
        ax: The matplotlib axes.
        location: String specifying where the arrow goes (e.g. 'lower left').
        scale: Size scale of the arrow.
        fontsize: Size of the 'N' label.
    """
    north_arrow(
        ax, 
        location=location, 
        rotation={"crs": 3857, "reference": "center"}, 
        fancy=True, 
        shadow=False, 
        scale=scale, 
        label={"position": "bottom", "text": "N", "fontsize": fontsize},
    )

# ==========================================
# 2. Geodesy & Coordinate Transformations
# ==========================================

def utm_2_latlon(utm_x: Union[float, np.ndarray], utm_y: Union[float, np.ndarray], source_epsg: str, dest_epsg: str = "EPSG:4326") -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    transformer = Transformer.from_crs(source_epsg, dest_epsg)
    return transformer.transform(utm_x, utm_y)

def latlon_2_utm(lat: Union[float, np.ndarray], lon: Union[float, np.ndarray], dest_epsg: str, source_epsg: str = "EPSG:4326") -> Tuple[Union[float, np.ndarray], Union[float, np.ndarray]]:
    transformer = Transformer.from_crs(source_epsg, dest_epsg)
    return transformer.transform(lat, lon)

def latlon_2_dist(lat1: float, lon1: float, lat2: float, lon2: float, method: str = "WGS84") -> float:
    if method == "WGS84":
        dist, _, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    elif method == "spherical":
        dist = degrees2kilometers(locations2degrees(lat1, lon1, lat2, lon2)) * 1e3
    return dist

def latlon_2_az(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, az, _ = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return az

def latlon_2_baz(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    _, _, baz = gps2dist_azimuth(lat1, lon1, lat2, lon2)
    return baz

# ==========================================
# 3. Geophysics Data Operations
# ==========================================

def projection(A: np.ndarray, B: np.ndarray, C: np.ndarray) -> np.ndarray:
    """Project point C onto vector AB."""
    A, B, C = np.array(A), np.array(B), np.array(C)
    
    if A.ndim == 1: A = A.reshape(1, -1)
    if B.ndim == 1: B = B.reshape(1, -1)
    if C.ndim == 1:
        C = C.reshape(1, -1)
        ndim = 1
    else:
        ndim = 2

    AB = B - A
    AC = C - A.reshape(1, -1)

    projection_length = np.sum(AC * AB, axis=1) / np.sum(AB**2)
    D = A + projection_length.reshape(-1, 1) * AB
    return D[0] if ndim == 1 else D


def vel2tz(depth: np.ndarray, vel: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute a two-way time-depth curve from instantaneous velocity vs depth."""
    if len(depth) != len(vel):
        raise ValueError("Depth and velocity arrays must have the same length")

    dt = np.diff(depth) / vel[:-1]
    t = np.cumsum(dt)
    t = np.insert(t, 0, 0)
    return 2 * t, depth

# ==========================================
# 4. 2D Inverted Model
# ==========================================

def plot_2d_contour_section(
    positions: Union[np.ndarray, List[float], List[int]], 
    z_grid: Union[np.ndarray, List[float], List[int]], 
    vs_2d_matrix: np.ndarray, 
    ax: Optional[plt.Axes] = None,          
    max_depth: Union[float, int] = 120, 
    vmin: Union[float, int] = 200, 
    vmax: Union[float, int] = 600, 
    levels: int = 50, 
    cmap: str = 'turbo', 
    figsize: Tuple[Union[float, int], Union[float, int]] = (12, 5), 
    smooth_sigma: Tuple[float, float] = (1, 2), 
    tick_step: int = 100, 
    contour: bool = False, 
    x_flip: bool = False,
    xlim: Optional[Tuple[float, float]] = None,    
    save_path: Optional[str] = None
):
    """
    Plots a 2D contoured shear-wave velocity cross-section.
    
    If an 'ax' is provided, it draws the data on that axis and returns the 
    contour object (useful for multi-panel GridSpec layouts). If no 'ax' is 
    provided, it acts as a standalone plotting function and generates its own 
    figure, labels, virtual shot markers, and colorbar.

    Returns:
        cf: The matplotlib QuadContourSet object (used for building colorbars).
    """
    # 1. Figure Setup
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
        created_fig = True
    else:
        created_fig = False

    # 2. Apply Smoothing
    if smooth_sigma != (0, 0):
        plot_matrix = gaussian_filter(vs_2d_matrix, sigma=smooth_sigma)
    else:
        plot_matrix = vs_2d_matrix

    # 3. Create Grid and Levels
    X, Z = np.meshgrid(positions, z_grid)
    contour_levels = np.linspace(vmin, vmax, levels)
    tick_levels = np.arange(vmin, vmax + tick_step, tick_step)
    
    # 4. Draw Filled Contours
    cf = ax.contourf(X, Z, plot_matrix, levels=contour_levels, cmap=cmap, extend='both')

    # 5. Draw Contour Lines (Optional)
    if contour:
        cl = ax.contour(X, Z, plot_matrix, levels=tick_levels, colors='black', linewidths=0.5, alpha=0.3)
        ax.clabel(cl, inline=True, fontsize=8, fmt='%1.0f')

    # 6. Core Axis Formatting
    ax.invert_yaxis()
    ax.set_ylim(max_depth, 0)
    
    if xlim is not None:
         ax.set_xlim(xlim)
    else:
        x_min, x_max = np.min(positions), np.max(positions)
        if x_flip:
            ax.set_xlim(x_max, x_min)
        else:
            ax.set_xlim(x_min, x_max)

    # 7. Standalone Mode Only (Skip if inside a larger GridSpec)
    if created_fig:
        # Add Virtual Shots
        ax.scatter(positions, np.zeros_like(positions), marker='v', color='black', 
                   s=50, clip_on=False, label='Virtual Shots', zorder=5)
        ax.legend(loc='upper left', fontsize=12)
        
        # Add Labels & Title
        ax.set_title("Contoured 2D Shear-Wave Velocity ($V_s$) Profile", fontsize=16, pad=15)
        ax.set_xlabel("Distance Along Cable (m)", fontsize=14)
        ax.set_ylabel("Depth (m)", fontsize=14)
        
        # Add Colorbar
        cbar = plt.colorbar(cf, ax=ax, pad=0.02, ticks=tick_levels)
        cbar.set_label('Shear-Wave Velocity ($V_s$) [m/s]', fontsize=12)

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure successfully saved to: {save_path}")
        plt.show()

    # Always return cf so master scripts can build floating colorbars!
    return cf

# ==========================================
# 4. Model Interpolation Classes
# ==========================================

class Model2D:
    """Class to store, grid, and visualize 2D spatially varying data."""
    def __init__(self, x: np.ndarray, z: np.ndarray, value: np.ndarray):
        self.init(x, z, value)

    def init(self, x: np.ndarray, z: np.ndarray, value: np.ndarray) -> None:
        sort_indices = np.lexsort((z, x))
        self.x, self.z, self.value = x[sort_indices], z[sort_indices], value[sort_indices]
        
        self.x_min, self.x_max = np.nanmin(x), np.nanmax(x)
        self.z_min, self.z_max = np.nanmin(z), np.nanmax(z)
        self.value_min, self.value_max = np.nanmin(value), np.nanmax(value)
        
        self.x_axis, self.x_index = np.unique(x, return_inverse=True)
        self.z_axis, self.z_index = np.unique(z, return_inverse=True)
        self.include_nan = np.isnan(self.value).any()

    def __str__(self) -> str:
        return (
            f"* Model2D: \n"
            f"            x_min: {self.x_min}\n"
            f"            x_max: {self.x_max}\n"
            f"            z_min: {self.z_min}\n"
            f"            z_max: {self.z_max}\n"
            f"        value_min: {self.value_min}\n"
            f"        value_max: {self.value_max}\n"
            f"        value_num: {len(self.value)} =? {len(self.x_axis)} * {len(self.z_axis)} || (nx_axis * nz_axis)\n"
        )
    
    def __repr__(self) -> str:
        return str(self)

    def save(self, filename: str) -> None:
        """Saves the 2D model arrays to a compressed .npz file."""
        np.savez(filename, x=self.x, z=self.z, value=self.value)

    def griddata(
        self,
        x_min: float, x_max: float, dx: float,
        z_min: float, z_max: float, dz: float,
        method: str = "linear",
        fill_value: float = np.nan,
        handle_nan: bool = True,
    ) -> None:
        """
        Interpolates scattered data onto a regular grid using scipy.interpolate.griddata.
        Modifies the model in place.
        """
        x = np.arange(x_min, x_max + dx, dx)
        z = np.arange(z_min, z_max + dz, dz)
        x_new, z_new = np.meshgrid(x, z, indexing="ij")
        points = np.vstack([self.x, self.z]).T

        value_new = griddata(
            points, self.value, (x_new, z_new), method=method, fill_value=fill_value
        )

        if np.isnan(value_new).any() and handle_nan:
            nan_indices = np.isnan(value_new)
            value_nearest = griddata(
                points, self.value,
                (x_new[nan_indices], z_new[nan_indices]), method="nearest"
            )
            value_new[nan_indices] = value_nearest

        self.init(x_new.flatten(), z_new.flatten(), value_new.flatten())

    def interpndata(
        self,
        x_min: float, x_max: float, dx: float,
        z_min: float, z_max: float, dz: float,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float = np.nan,
    ) -> None:
        """
        Interpolates regular rectilinear data onto a new grid using scipy.interpolate.interpn.
        Note: The existing data MUST be on a strict rectilinear grid for this to work.
        """
        x = np.arange(x_min, x_max + dx, dx)
        z = np.arange(z_min, z_max + dz, dz)
        grid_x, grid_z = np.meshgrid(x, z, indexing="ij")

        points = (self.x_axis, self.z_axis)
        values = self.value.reshape((len(self.x_axis), len(self.z_axis)))

        value_new = interpn(
            points, values, (grid_x, grid_z),
            method=method, bounds_error=bounds_error, fill_value=fill_value
        )

        self.init(grid_x.flatten(), grid_z.flatten(), value_new.flatten())

    def layer(
        self,
        x: float,
        z_num: int = 100,
        method: str = "linear",
        plot: bool = False,
        figsize: Tuple[float, float] = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts a 1D vertical depth profile at a specific x-coordinate."""
        z_axis = np.linspace(self.z_min, self.z_max, z_num)
        grid_x, grid_z = np.meshgrid([x], z_axis, indexing="ij")

        points = (self.x_axis, self.z_axis)
        values = self.value.reshape((len(self.x_axis), len(self.z_axis)))
        z_values = interpn(
            points, values, (grid_x, grid_z),
            method=method, bounds_error=False, fill_value=None
        ).flatten()

        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(z_values, z_axis, "o-", color="red")
            ax.set_xlabel("Value")
            ax.set_ylabel("Z Axis")
            ax.invert_yaxis()
            ax.grid()

            if save_path is not None:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        return z_values, z_axis

    def plot(
        self,
        cmap: str = "jet_r",
        clip: Tuple[Optional[float], Optional[float]] = (None, None),
        figsize: Tuple[float, float] = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ) -> None:
        """Visualizes the 2D model as an image with colorbar mapping."""
        fig, ax = plt.subplots(figsize=figsize)
        values = self.value.reshape((len(self.x_axis), len(self.z_axis)))
        im = ax.imshow(
            values.T,
            origin="lower",
            aspect="auto",
            cmap=cmap,
            extent=[self.x_min, self.x_max, self.z_min, self.z_max],
        )

        if clip[0] is not None and clip[1] is not None:
            im.set_clim(clip)

        ax.invert_yaxis()
        ax.set_xlabel("X Axis")
        ax.set_ylabel("Z Axis")
        cbar = plt.colorbar(im, orientation="vertical", ax=ax)
        cbar.set_label("Value")

        if save_path is not None:
            fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
        if show:
            plt.show()
        else:
            plt.close(fig)


class Model3D:
    """Class to store, grid, slice, and visualize 3D spatially varying data."""
    
    def __init__(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, value: np.ndarray):
        self.init(x, y, z, value)

    def init(self, x: np.ndarray, y: np.ndarray, z: np.ndarray, value: np.ndarray) -> None:
        """Initializes or resets the 3D model with new coordinates and data."""
        sort_indices = np.lexsort((z, y, x))  # sort by x, y, z
        self.x, self.y, self.z = x[sort_indices], y[sort_indices], z[sort_indices]
        self.value = value[sort_indices]
        
        self.x_min, self.x_max = np.nanmin(x), np.nanmax(x)
        self.y_min, self.y_max = np.nanmin(y), np.nanmax(y)
        self.z_min, self.z_max = np.nanmin(z), np.nanmax(z)
        self.value_min, self.value_max = np.nanmin(value), np.nanmax(value)
        
        self.x_axis, self.x_index = np.unique(x, return_inverse=True)
        self.y_axis, self.y_index = np.unique(y, return_inverse=True)
        self.z_axis, self.z_index = np.unique(z, return_inverse=True)
        self.include_nan = np.isnan(self.value).any()

    def __str__(self) -> str:
        return (
            f"* Model3D: \n"
            f"            x_min: {self.x_min}\n"
            f"            x_max: {self.x_max}\n"
            f"            y_min: {self.y_min}\n"
            f"            y_max: {self.y_max}\n"
            f"            z_min: {self.z_min}\n"
            f"            z_max: {self.z_max}\n"
            f"      include_nan: {self.include_nan}\n"
            f"        value_min: {self.value_min}\n"
            f"        value_max: {self.value_max}\n"
            f"        value_num: {len(self.value)} =? {len(self.x_axis)} * {len(self.y_axis)} * {len(self.z_axis)}\n"
        )

    def __repr__(self) -> str:
        return str(self)

    def save(self, filename: str) -> None:
        """Saves the 3D model arrays to a compressed .npz file."""
        np.savez(filename, x=self.x, y=self.y, z=self.z, value=self.value)

    def griddata(
        self,
        x_min: float, x_max: float, dx: float,
        y_min: float, y_max: float, dy: float,
        z_min: float, z_max: float, dz: float,
        method: str = "linear",
        fill_value: float = np.nan,
        handle_nan: bool = True,
    ) -> None:
        """Interpolates scattered 3D data onto a regular 3D grid."""
        x = np.arange(x_min, x_max + dx, dx)
        y = np.arange(y_min, y_max + dy, dy)
        z = np.arange(z_min, z_max + dz, dz)

        x_new, y_new, z_new = np.meshgrid(x, y, z, indexing="ij")
        points = np.vstack([self.x, self.y, self.z]).T

        value_new = griddata(
            points, self.value, (x_new, y_new, z_new),
            method=method, fill_value=fill_value
        )

        if np.isnan(value_new).any() and handle_nan:
            nan_indices = np.isnan(value_new)
            value_nearest = griddata(
                points, self.value,
                (x_new[nan_indices], y_new[nan_indices], z_new[nan_indices]), method="nearest"
            )
            value_new[nan_indices] = value_nearest

        self.init(x_new.flatten(), y_new.flatten(), z_new.flatten(), value_new.flatten())

    def interpndata(
        self,
        x_min: float, x_max: float, dx: float,
        y_min: float, y_max: float, dy: float,
        z_min: float, z_max: float, dz: float,
        method: str = "linear",
        bounds_error: bool = False,
        fill_value: float = np.nan,
    ) -> None:
        """Interpolates regular rectilinear 3D data onto a new grid."""
        x = np.arange(x_min, x_max + dx, dx)
        y = np.arange(y_min, y_max + dy, dy)
        z = np.arange(z_min, z_max + dz, dz)
        grid_x, grid_y, grid_z = np.meshgrid(x, y, z, indexing="ij")

        points = (self.x_axis, self.y_axis, self.z_axis)
        values = self.value.reshape((len(self.x_axis), len(self.y_axis), len(self.z_axis)))

        value_new = interpn(
            points, values, (grid_x, grid_y, grid_z),
            method=method, bounds_error=bounds_error, fill_value=fill_value
        )

        self.init(grid_x.flatten(), grid_y.flatten(), grid_z.flatten(), value_new.flatten())

    def layer(
        self,
        point: Tuple[float, float],
        mode: str = "num",
        dz: float = 10.0,
        z_num: int = 100,
        method: str = "linear",
        plot: bool = False,
        figsize: Tuple[float, float] = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts a 1D vertical depth profile at a specific (x, y) coordinate."""
        x_point, y_point = point

        if mode == "num":
            z_axis = np.linspace(self.z_min, self.z_max, z_num)
        elif mode == "interval":
            z_axis = np.arange(self.z_min, self.z_max, dz)
            
        grid_x, grid_y, grid_z = np.meshgrid([x_point], [y_point], z_axis, indexing="ij")

        points = (self.x_axis, self.y_axis, self.z_axis)
        values = self.value.reshape((len(self.x_axis), len(self.y_axis), len(self.z_axis)))
        
        z_values = interpn(
            points, values, (grid_x, grid_y, grid_z),
            method=method, bounds_error=False, fill_value=None
        ).flatten()

        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            ax.plot(z_values, z_axis, "o-", color="red")
            ax.set_xlabel("Value")
            ax.set_ylabel("Z Axis")
            ax.invert_yaxis()
            ax.grid()

            if save_path is not None:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        return z_values, z_axis

    def slice(
        self,
        z: Union[float, np.ndarray],
        mode: str = "num",
        dx: float = 10.0,
        dy: float = 10.0,
        x_num: int = 100,
        y_num: int = 100,
        method: str = "linear",
        fill_value: float = np.nan,
        plot: bool = False,
        cmap: str = "jet_r",
        clip: Tuple[Optional[float], Optional[float]] = (None, None),
        figsize: Tuple[float, float] = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts and plots a 2D horizontal slice at a specific depth (z)."""
        if mode == "num":
            x_axis = np.linspace(self.x_min, self.x_max, x_num)
            y_axis = np.linspace(self.y_min, self.y_max, y_num)
        elif mode == "interval":
            x_axis = np.arange(self.x_min, self.x_max, dx)
            y_axis = np.arange(self.y_min, self.y_max, dy)
            
        grid_x, grid_y, grid_z = np.meshgrid(x_axis, y_axis, z, indexing="ij")

        points = (self.x_axis, self.y_axis, self.z_axis)
        values = self.value.reshape((len(self.x_axis), len(self.y_axis), len(self.z_axis)))
        
        slice_values = interpn(
            points, values, (grid_x, grid_y, grid_z),
            method=method, bounds_error=False, fill_value=fill_value
        )[:, :, 0]

        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                slice_values.T, origin="lower", aspect="auto", cmap=cmap,
                extent=[self.x_min, self.x_max, self.y_min, self.y_max]
            )

            if clip[0] is not None and clip[1] is not None:
                im.set_clim(clip)

            ax.set_xlabel("X Axis")
            ax.set_ylabel("Y Axis")
            cbar = plt.colorbar(im, orientation="vertical", ax=ax)
            cbar.set_label("Value")

            if save_path is not None:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        return slice_values, x_axis, y_axis

    def profile(
        self,
        point1: Tuple[float, float],
        point2: Tuple[float, float],
        mode: str = "num",
        d_dist: float = 10.0,
        dz: float = 10.0,
        dist_num: int = 100,
        z_num: int = 100,
        method: str = "linear",
        fill_value: float = np.nan,
        plot: bool = False,
        cmap: str = "jet_r",
        clip: Tuple[Optional[float], Optional[float]] = (None, None),
        figsize: Tuple[float, float] = (10, 6),
        show: bool = True,
        save_path: Optional[str] = None,
        dpi: int = 100,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Extracts and plots a 2D vertical cross-section between two (x, y) coordinates."""
        x1, y1 = point1
        x2, y2 = point2

        if mode == "num":
            x_axis = np.linspace(x1, x2, dist_num)
            y_axis = np.linspace(y1, y2, dist_num)
            z_axis = np.linspace(self.z_min, self.z_max, z_num)
            distances = np.sqrt((x_axis - x1) ** 2 + (y_axis - y1) ** 2)
        elif mode == "interval":
            theta = np.arctan2(y2 - y1, x2 - x1)
            distances = np.arange(0, np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2), d_dist)
            x_axis = x1 + distances * np.cos(theta)
            y_axis = y1 + distances * np.sin(theta)
            z_axis = np.arange(self.z_min, self.z_max, dz)

        grid_x, grid_y, grid_z = np.meshgrid(x_axis, y_axis, z_axis, indexing="ij")

        points = (self.x_axis, self.y_axis, self.z_axis)
        values = self.value.reshape((len(self.x_axis), len(self.y_axis), len(self.z_axis)))
        
        profile_values = interpn(
            points, values, (grid_x[:, 0, :], grid_y[0, :, :], grid_z[0, 0, :]),
            method=method, bounds_error=False, fill_value=fill_value
        )

        if plot:
            fig, ax = plt.subplots(figsize=figsize)
            im = ax.imshow(
                profile_values.T, origin="lower", aspect="auto", cmap=cmap,
                extent=[0, distances[-1], self.z_min, self.z_max]
            )

            if clip[0] is not None and clip[1] is not None:
                im.set_clim(clip)

            ax.invert_yaxis()
            ax.set_xlabel("Distance along the Profile")
            ax.set_ylabel("Z Axis")
            cbar = plt.colorbar(im, orientation="vertical", ax=ax)
            cbar.set_label("Value")

            if save_path is not None:
                fig.savefig(save_path, dpi=dpi, bbox_inches='tight')
            if show:
                plt.show()
            else:
                plt.close(fig)

        return profile_values, distances, z_axis