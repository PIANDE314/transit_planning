import osmnx as ox
import geopandas as gpd
from typing import Tuple
import networkx as nx

"""
§ 3 Passive Datasets — OSM road network and POIs :contentReference[oaicite:3]{index=3}
"""

def load_osm_network(
    place_name: str,
    network_type: str = "drive"
) -> Tuple[nx.Graph, nx.Graph]:
    """
    Download and project the OSM road network for a place.

    Returns both:
      - G_latlon: lat/lon graph (EPSG:4326)
      - G_proj:   projected graph (EPSG:3857)

    Args:
        place_name: e.g., "Maputo, Mozambique".
        network_type: OSMnx network type, default "drive".

    See § 3 “Extract OSM driving network via OSMnx” :contentReference[oaicite:4]{index=4}
    """
    G_latlon = ox.graph_from_place(place_name, network_type=network_type)
    G_proj   = ox.project_graph(G_latlon, to_crs="EPSG:3857")
    return G_latlon, G_proj


def load_osm_pois(
    place_name: str,
    tags: dict = None
) -> gpd.GeoDataFrame:
    """
    Download and filter OSM POIs relevant to bus infrastructure.

    Defaults to:
      highway=bus_stop, public_transport=[platform, station], amenity=bus_station

    Note: POIs are loaded here; the 100 m “within 100 m buffer” for
    stop‑infrastructure influence is applied downstream (§ 3, Fig 2B). :contentReference[oaicite:5]{index=5}

    Returns POIs in EPSG:3857.

    Args:
        place_name: same as load_osm_network.
        tags: OSM feature tags to filter.
    """
    if tags is None:
        tags = {
            "highway": ["bus_stop"],
            "public_transport": ["platform", "station"],
            "amenity": ["bus_station"]
        }
    pois = ox.features_from_place(place_name, tags=tags)
    pois = pois[pois.geometry.geom_type == "Point"]
    return pois.to_crs(epsg=3857)
