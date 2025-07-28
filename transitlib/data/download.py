import requests
import zipfile
from pathlib import Path
from typing import Union, Optional

import rasterio
from rasterio.windows import from_bounds
from shapely.geometry import mapping
from transitlib.config import Config

cfg = Config()

def download_file(
    url: str,
    dest: Union[str, Path],
    overwrite: bool = False,
    timeout: Optional[int] = None
) -> Path:
    """
    Download a file with streaming. If dest exists (and not overwrite), skip.
    Performs a HEAD first to verify URL if dest missing.
    """
    timeout = timeout or cfg.get("download_timeout", 10)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    # verify URL is reachable
    head = requests.head(url, timeout=timeout)
    head.raise_for_status()

    # stream download
    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    return dest


def fetch_worldpop_cog_crop(
    cog_url: str,
    region_geom,
    dest_tif: Union[str, Path]
) -> Path:
    """
    Crop a WorldPop Cloud‑Optimized GeoTIFF (COG) directly over HTTP
    to the bounds of `region_geom`, writing only that window to `dest_tif`.
    """
    dest_tif = Path(dest_tif)
    dest_tif.parent.mkdir(parents=True, exist_ok=True)

    # Use vsicurl to stream only the bytes needed
    vsicurl_path = f"/vsicurl/{cog_url}"

    with rasterio.open(vsicurl_path) as src:
        minx, miny, maxx, maxy = region_geom.bounds
        window = from_bounds(minx, miny, maxx, maxy, src.transform)
        profile = src.profile.copy()
        profile.update({
            "driver": "GTiff",
            "height": window.height,
            "width":  window.width,
            "transform": src.window_transform(window)
        })
        data = src.read(1, window=window)

    with rasterio.open(dest_tif, "w", **profile) as dst:
        dst.write(data, 1)

    return dest_tif


def worldpop_stats(
    region_geom,
    dataset: str = "wpgppop"
) -> dict:
    """
    Query WorldPop's REST 'stats' API for aggregate population over a GeoJSON region.
    Returns a dict containing 'sum', 'mean', 'total', etc.
    """
    url = "https://www.worldpop.org/rest/data/stats"
    payload = {"dataset": dataset, "geom": mapping(region_geom)}
    resp = requests.post(url, json=payload, timeout=cfg.get("download_timeout", 10))
    resp.raise_for_status()
    return resp.json()


def fetch_hdx_rwi_csv(
    *,
    manual_csv: Union[str, Path] = None,
    manual_url: str = None,
    country_name: str = None,
    country_code: str = None,
    dest: Union[str, Path]
) -> Path:
    """
    Retrieve the Relative Wealth Index CSV.
    - If `manual_csv` is supplied, verify and return it.
    - Else if `manual_url` is given, download that.
    - Otherwise search HDX for "{country_name} relative wealth index"
      and pick the first resource ending in "relative_wealth_index.csv".
    """
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if manual_csv:
        mc = Path(manual_csv)
        if not mc.exists():
            raise FileNotFoundError(f"Manual RWI CSV not found: {mc}")
        return mc

    if manual_url:
        url = manual_url
    else:
        cn = country_name or cfg.get("country_name")
        cc = (country_code or cfg.get("country_code")).lower()
        search_api = cfg.get("hdx_search_api")
        show_api   = cfg.get("hdx_show_api")

        # 1) search
        resp = requests.get(search_api, params={"q": f"{cn} relative wealth index"})
        resp.raise_for_status()
        results = resp.json()["result"]["results"]
        if not results:
            raise RuntimeError(f"No HDX RWI dataset for '{cn}'")
        ds_id = results[0]["id"]

        # 2) show
        resp = requests.get(show_api, params={"id": ds_id})
        resp.raise_for_status()
        resources = resp.json()["result"]["resources"]

        # 3) pick resource by suffix or fallback
        suffix = "relative_wealth_index.csv"
        candidates = [
            r["url"] for r in resources
            if r.get("url","").lower().endswith(suffix)
        ]
        if not candidates:
            candidates = [
                r["url"] for r in resources
                if suffix in r.get("url","").lower()
            ]
        if not candidates:
            raise RuntimeError(f"No RWI CSV found for '{cn}'")
        url = candidates[0]

    # download to dest
    return download_file(url, dest)
