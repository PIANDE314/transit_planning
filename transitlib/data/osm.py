import osmnx as ox
import geopandas as gpd
from typing import Tuple, Dict
import networkx as nx
import pickle
from pathlib import Path
from transitlib.config import Config

cfg = Config()

def _save_graph_pickle(G, path: Path):
    """Save a NetworkX graph (with all attrs) to a .pkl file."""
    with open(path, "wb") as f:
        pickle.dump(G, f, protocol=pickle.HIGHEST_PROTOCOL)

def _load_graph_pickle(path: Path):
    """Load a pickled NetworkX graph."""
    with open(path, "rb") as f:
        return pickle.load(f)

def load_osm_network(
    place_name: str,
    network_type: str = None
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Download and project the OSM road network for a place.
    """
    network_type = network_type or cfg.get("osm_network_type", "drive")
    raw_dir = Path(cfg.get("raw_data_dir"))
    raw_dir.mkdir(parents=True, exist_ok=True)

    # filenames based on place
    safe = place_name.replace(" ", "_")
    lat_fn  = raw_dir / f"{safe}_network_latlon.pkl"
    proj_fn = raw_dir / f"{safe}_network_proj.pkl"

    # load if already saved
    if lat_fn.exists() and proj_fn.exists():
        G_latlon = _load_graph_pickle(lat_fn)
        G_proj   = _load_graph_pickle(proj_fn)
        return G_latlon, G_proj
    
    # otherwise fetch once and save
    G_latlon = ox.graph_from_place(place_name, network_type=network_type)
    G_proj   = ox.project_graph(G_latlon, to_crs="EPSG:3857")

    # persist for next time
    _save_graph_pickle(G_latlon,  lat_fn)
    _save_graph_pickle(G_proj,   proj_fn)
    return G_latlon, G_proj


def load_osm_pois(
    place_name: str,
    tags: Dict[str, list] = None
) -> gpd.GeoDataFrame:
    """
    Download and filter OSM POIs relevant to bus infrastructure.
    """
    tags = tags or cfg.get("osm_poi_tags", {
        "highway": ["bus_stop"],
        "public_transport": ["platform", "station"],
        "amenity": ["bus_station"]
    })
    pois = ox.features_from_place(place_name, tags=tags)
    pois = pois[pois.geometry.geom_type == "Point"]
    return pois.to_crs(epsg=3857)
