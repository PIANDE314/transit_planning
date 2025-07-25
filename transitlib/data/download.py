import os
from pathlib import Path
import requests
import zipfile
from typing import Union
from transit_planner.config import Config

cfg = Config()

"""
§ 3 Passive Datasets — Download helpers for WorldPop and HDX RWI
"""

def download_file(
    url: str,
    dest: Union[str, Path],
    overwrite: bool = False,
    timeout: int = None
) -> Path:
    """
    Download a file with streaming and optional overwrite.
    """
    timeout = timeout or cfg.get("download_timeout", 10)
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)

    if dest.exists() and not overwrite:
        return dest

    resp = requests.get(url, stream=True, timeout=timeout)
    resp.raise_for_status()

    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)

    return dest


def fetch_worldpop_url(
    country_code: str,
    pop_version: str = None
) -> str:
    """
    Build WorldPop download URL.
    """
    pop_version = pop_version or cfg.get("pop_version")
    v_ = pop_version.replace(".", "_")
    base = f"https://data.worldpop.org/repo/wopr/{country_code}/population/{pop_version}"
    return f"{base}/{country_code}_population_{v_}_gridded.zip"


def extract_worldpop_tif(
    zip_path: Union[str, Path],
    dest_dir:   Union[str, Path]
) -> Path:
    """
    Unzip and locate the first .tif inside.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith('.tif'):
                z.extract(name, dest_dir)
                return dest_dir / name

    raise FileNotFoundError(f"No .tif found in {zip_path}")


def fetch_hdx_rwi_url(
    country_name: str,
    country_code: str
) -> str:
    """
    Search HDX for Relative Wealth Index CSV.
    """
    search_api = cfg.get("hdx_search_api", "https://data.humdata.org/api/3/action/package_search")
    show_api   = cfg.get("hdx_show_api",   "https://data.humdata.org/api/3/action/package_show")

    resp = requests.get(search_api, params={"q": f"{country_name} relative wealth index"})
    resp.raise_for_status()
    results = resp.json()["result"]["results"]
    if not results:
        raise RuntimeError(f"No HDX RWI dataset for '{country_name}'")

    ds_id = results[0]["id"]
    resp = requests.get(show_api, params={"id": ds_id})
    resp.raise_for_status()
    resources = resp.json()["result"]["resources"]

    suffix = f"{country_code.lower()}_relative_wealth_index.csv"
    for res in resources:
        url = res.get("url", "")
        if url.lower().endswith(suffix):
            return url

    raise RuntimeError(f"No RWI CSV found ending with '{suffix}'")
