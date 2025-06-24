import math
from PIL import Image
import requests
from io import BytesIO

TILE_SIZE = 256
def fetch_tile(x, y, zoom):
    url = f"https://tile.openstreetmap.org/{zoom}/{x}/{y}.png"
    headers = {'User-Agent': 'dispersion_modelling/0.1'}
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return Image.open(BytesIO(resp.content)).convert("RGBA")

def point_to_pixels(lon, lat, zoom):
    r = 2 ** zoom * TILE_SIZE
    lat_rad = math.radians(lat)
    x = (lon + 180.0) / 360.0 * r
    y = (1.0 - math.log(math.tan(lat_rad) + 1.0 / math.cos(lat_rad)) / math.pi) / 2.0 * r
    return x, y

def pixels_to_point(x, y, zoom):
    """Convert pixel coordinates to (longitude, latitude) at given zoom level."""
    r = 2 ** zoom * TILE_SIZE
    lon = x / r * 360.0 - 180.0

    n = math.pi - 2.0 * math.pi * y / r
    lat = math.degrees(math.atan(math.sinh(n)))
    
    return lon, lat