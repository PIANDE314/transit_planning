import os
from pathlib import Path
import requests
import zipfile
from typing import Union

"""
§ 3 Passive Datasets — Download helpers for WorldPop and HDX RWI :contentReference[oaicite:0]{index=0}
"""

def download_file(
    url: str,
    dest: Union[str, Path],
    overwrite: bool = False,
    timeout: int = 10
) -> Path:
    """
    Download a file with streaming and optional overwrite.

    Args:
        url: HTTP URL of the file.
        dest: Local path to save.
        overwrite: If False and file exists, skip download.
        timeout: Seconds before HTTP timeout.

    Returns:
        Path to the downloaded (or existing) file.
    """
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
    pop_version: str = "v2.1"
) -> str:
    """
    Build WorldPop download URL for gridded population.

    Paper § 3: “Gridded population data from WorldPop (v2.1) at 100 m resolution” :contentReference[oaicite:1]{index=1}

    Args:
        country_code: ISO3 country code (e.g., "MOZ").
        pop_version: version string (default "v2.1").

    Returns:
        URL to the zipped GeoTIFF.
    """
    v_ = pop_version.replace(".", "_")
    base = f"https://data.worldpop.org/repo/wopr/{country_code}/population/{pop_version}"
    return f"{base}/{country_code}_population_{v_}_gridded.zip"


def extract_worldpop_tif(
    zip_path: Union[str, Path],
    dest_dir:   Union[str, Path]
) -> Path:
    """
    Unzip a WorldPop ZIP archive and locate the first .tif inside.

    Args:
        zip_path: Path to downloaded .zip.
        dest_dir: Directory to extract into.

    Returns:
        Path to the extracted .tif file.
    """
    zip_path = Path(zip_path)
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        for name in z.namelist():
            if name.lower().endswith('.tif'):
                z.extract(name, dest_dir)
                tif_path = dest_dir / name
                return tif_path

    raise FileNotFoundError(f"No .tif found in {zip_path}")


def fetch_hdx_rwi_url(
    country_name: str,
    country_code: str
) -> str:
    """
    Search HDX for Relative Wealth Index CSV.

    Paper § 3: “Relative Wealth Index (RWI) from HDX” :contentReference[oaicite:2]{index=2}

    Args:
        country_name: Full country name (e.g., "Mozambique").
        country_code: ISO3 code lowercased (e.g., "moz").

    Returns:
        Direct download URL for the RWI CSV.
    """
    search_api = "https://data.humdata.org/api/3/action/package_search"
    show_api   = "https://data.humdata.org/api/3/action/package_show"

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
