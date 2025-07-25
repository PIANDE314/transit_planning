import osmnx as ox
import geopandas as gpd
from typing import Tuple, Dict
import networkx as nx
from transit_planner.config import Config

cfg = Config()

"""
§ 3 Passive Datasets — OSM road network and POIs
"""

def load_osm_network(
    place_name: str,
    network_type: str = None
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Download and project the OSM road network for a place.
    """
    network_type = network_type or cfg.get("osm_network_type", "drive")
    G_latlon = ox.graph_from_place(place_name, network_type=network_type)
    G_proj   = ox.project_graph(G_latlon, to_crs="EPSG:3857")
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
