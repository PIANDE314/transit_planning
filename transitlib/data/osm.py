import pickle
from pathlib import Path
from typing import Union, Dict, Tuple

import osmnx as ox
import geopandas as gpd
import networkx as nx


def load_osm_network(
    *,
    place_name: str,
    raw_dir: Union[str, Path],
    manual_graph_latlon: Union[str, Path] = None,
    manual_graph_proj:  Union[str, Path] = None,
    network_type:       str = "drive"
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Load or fetch the OSM road network.
    - If both `manual_graph_latlon` and `manual_graph_proj` exist, load them.
    - Otherwise geocode `place_name`, fetch via Overpass, save pickles to `raw_dir`.
    """
    raw_dir = Path(raw_dir); raw_dir.mkdir(parents=True, exist_ok=True)
    safe = place_name.replace(" ", "_")

    lat_fn = Path(manual_graph_latlon) if manual_graph_latlon \
             else raw_dir / f"{safe}_network_latlon.pkl"
    proj_fn = Path(manual_graph_proj) if manual_graph_proj \
              else raw_dir / f"{safe}_network_proj.pkl"

    # 1) load existing if available
    if lat_fn.exists() and proj_fn.exists():
        with open(lat_fn,  "rb") as f: G_latlon = pickle.load(f)
        with open(proj_fn, "rb") as f: G_proj   = pickle.load(f)
        return G_latlon, G_proj

    # 2) verify geocoding
    try:
        gdf = ox.geocode_to_gdf(place_name)
        if gdf.empty:
            raise ValueError("empty result")
    except Exception as e:
        raise RuntimeError(f"Place '{place_name}' not found by Nominatim: {e}")

    # 3) fetch network via Overpass
    G_latlon = ox.graph_from_place(place_name, network_type=network_type)
    G_proj   = ox.project_graph(G_latlon, to_crs="EPSG:3857")

    # 4) save for next time
    with open(lat_fn,  "wb") as f: pickle.dump(G_latlon, f)
    with open(proj_fn, "wb") as f: pickle.dump(G_proj,   f)

    return G_latlon, G_proj


def load_osm_pois(
    *,
    place_name: str,
    raw_dir: Union[str, Path],
    manual_pois: Union[str, Path] = None,
    tags: Dict[str, list] = None
) -> gpd.GeoDataFrame:
    """
    Load or fetch OSM POIs for bus infrastructure.
    - If `manual_pois` exists, read it.
    - Otherwise query Overpass via OSMnx.features_from_place.
    """
    if manual_pois:
        pois = gpd.read_file(manual_pois)
        if pois.empty or pois.geometry.isna().any():
            raise RuntimeError(f"Invalid manual POIs file: {manual_pois}")
        return pois.to_crs(epsg=3857)

    # fetch
    tags = tags or {
        "highway": ["bus_stop"],
        "public_transport": ["platform", "station"],
        "amenity": ["bus_station"]
    }
    try:
        pois = ox.features_from_place(place_name, tags=tags)
    except Exception as e:
        raise RuntimeError(f"Error fetching POIs for '{place_name}': {e}")

    pois = pois[pois.geometry.geom_type == "Point"]
    return pois.to_crs(epsg=3857)
