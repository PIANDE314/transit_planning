import pandas as pd
import geopandas as gpd
import rasterio
from rasterio.mask import mask
from shapely.geometry.base import BaseGeometry
from pathlib import Path
from typing import Union

def clip_raster_to_region(
    raster_path:   Union[str, Path],
    region_geom:   BaseGeometry,
    out_path:      Union[str, Path]
) -> Path:
    """
    Clip a GeoTIFF to the given region geometry.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(exist_ok=True, parents=True)

    with rasterio.open(raster_path) as src:
        img, transform = mask(src, [region_geom], crop=True)
        meta = src.meta.copy()
        meta.update({
            "driver": "GTiff",
            "height": img.shape[1],
            "width": img.shape[2],
            "transform": transform
        })
        with rasterio.open(out_path, "w", **meta) as dst:
            dst.write(img)

    return out_path


def load_rwi_csv(
    csv_path: Union[str, Path]
) -> pd.DataFrame:
    """
    Read the Relative Wealth Index CSV into a DataFrame.
    """
    return pd.read_csv(csv_path)


def points_to_gdf(
    df:    pd.DataFrame,
    x_col: str,
    y_col: str,
    crs:    str = "EPSG:4326"
) -> gpd.GeoDataFrame:
    """
    Convert a DataFrame with lon/lat columns to a GeoDataFrame.
    """
    gdf = gpd.GeoDataFrame(
        df,
        geometry=gpd.points_from_xy(df[x_col], df[y_col]),
        crs=crs
    )
    return gdf.to_crs(epsg=3857)
