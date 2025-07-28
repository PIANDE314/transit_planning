import requests
from pathlib import Path
from typing import Union, Optional

import rasterio
from rasterio.mask import mask
from rasterio.vrt import WarpedVRT
from rasterio.enums import Resampling
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
    place_name: str,
    country_code: str,
    pop_version: str,
    dest_dir: Path,
    region_geom
) -> Path:
    """
    Stream a COG from WorldPop and crop it to a city region (no full download).
    """
    # Build the /vsigs/ path to the public WorldPop COG (100 m PPP)
    year = pop_version
    vsigs_path = (
        f"/vsigs/gcp-public-data-worldpop/GIS/Population/Global_2000_2020/"
        f"{year}/{country_code.upper()}_ppp_{year}.tif"
    )

    # Prepare output
    dest_dir.mkdir(parents=True, exist_ok=True)
    safe_name = place_name.replace(" ", "_").replace(",", "")
    out_tif = dest_dir / f"worldpop_{safe_name}_{pop_version}.tif"

    # GeoJSON geometry for masking
    geom_json = [mapping(region_geom)]

    # Tell GDAL not to try any OAuth2 or signed requests—this bucket is public.
    with rasterio.Env(
        GDAL_DISABLE_READDIR_ON_OPEN="YES",
        CPL_GS_NO_SIGN_REQUEST="TRUE"
    ):
        # Open the remote COG; Rasterio will use HTTP Range requests
        with rasterio.open(vsigs_path) as src:
            # Warp if needed and crop in one go
            with WarpedVRT(src, resampling=Resampling.nearest) as vrt:
                out_image, out_transform = mask(vrt, geom_json, crop=True)
                out_meta = vrt.meta.copy()
                out_meta.update({
                    "driver":   "GTiff",
                    "height":   out_image.shape[1],
                    "width":    out_image.shape[2],
                    "transform":out_transform,
                    "count":    out_image.shape[0]
                })

                # Write the clipped GeoTIFF
                with rasterio.open(out_tif, "w", **out_meta) as dst:
                    dst.write(out_image)

    return out_tif

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
