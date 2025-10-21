import matplotlib.pyplot as plt
import numpy as np
import skyproj
from rubin_scheduler.scheduler.utils import get_current_footprint

__all__ = ("get_background", "make_plot")


def get_background(nside=64):
    fp, labels = get_current_footprint(nside=nside)
    bg_fp = np.where(fp["r"] == 0, np.nan, fp["r"])
    bg_fp = np.where(bg_fp > 1, 1, bg_fp)
    return bg_fp


def make_plot(
    metric_bundle, proj="laea", vmin=None, vmax=None, ax=None, background=None, title=None, label_dec=True
):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))
    else:
        fig = ax.get_figure()

    if proj == "laea":
        sp = skyproj.LaeaSkyproj(
            ax=ax,
            celestial=True,
            galactic=False,
            gridlines=True,
            n_grid_lon=8,
            n_grid_lat=5,
            lat_0=-90,
            lon_0=0,
            extent=[0.0, 360.0, -90, 85],
        )
        # Laea only shows half the sky if zoom = True
        # due to bug in current skyproj
        zoom = False
    else:
        sp = skyproj.McBrydeSkyproj(
            ax=ax,
            celestial=True,
            galactic=False,
            gridlines=True,
            n_grid_lon=8,
            n_grid_lat=7,
            lon_0=0,
        )
        zoom = True

    if fig is not None and proj == "laea":
        sp.ax.set_xlabel("R.A.", fontsize=12, labelpad=9)
        if label_dec:
            sp.ax.set_ylabel("Dec.", fontsize=12, labelpad=12)

    if background is not None:
        mesh, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(
            background, cmap="Greys", vmin=-1, vmax=4, nest=False, zoom=False, zorder=0
        )

        zoom = False

    mesh, lon_raster, lat_raster, values_raster = sp.draw_hpxmap(
        metric_bundle.metric_values.filled(np.nan), vmin=vmin, vmax=vmax, nest=False, zoom=zoom, zorder=1.5
    )

    if vmin is None and vmax is None:
        extend = None
    elif vmin is None and vmax is not None:
        extend = "max"
    elif vmin is not None and vmax is None:
        extend = "min"
    else:
        extend = "both"
    _ = sp.draw_colorbar(
        label=f"{metric_bundle.metric.name} {metric_bundle.info_label}",
        pad=0.12,
        shrink=0.5,
        extend=extend,
        location="bottom",
        orientation="horizontal",
    )

    if proj == "laea":
        pass
    else:
        sp.ax.set_xlabel("R.A.", fontsize=12, labelpad=5)
        if label_dec:
            sp.ax.set_ylabel("Dec.", fontsize=12, labelpad=10)
        else:
            sp.ax.set_ylabel(None)

    if title is not None:
        plt.title(title, fontsize="x-large", pad=30)
    return fig
