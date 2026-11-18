"""Shared per-trip metric geometry (equirectangular projection)."""
import numpy as np
EARTH_R = 6_371_000.0

def project(lon, lat, lon0, lat0):
    return (EARTH_R * np.radians(np.asarray(lon)-lon0) * np.cos(np.radians(lat0)), EARTH_R * np.radians(np.asarray(lat)-lat0))

def add_xy(trip):
    """Add projected x/y to both frames, centered at the mean track coordinate."""
    lon0,lat0=trip.track.lon.mean(),trip.track.lat.mean()
    for frame in (trip.track,trip.nodes):
        frame['x'],frame['y']=project(frame.lon,frame.lat,lon0,lat0)
    return trip

def point_segment_distance(px,py,ax,ay,bx,by):
    dx=np.asarray(bx)-ax; dy=np.asarray(by)-ay; denom=dx*dx+dy*dy
    num=(np.asarray(px)-ax)*dx+(np.asarray(py)-ay)*dy
    t=np.clip(np.divide(num,denom,out=np.zeros_like(np.asarray(num),dtype=float),where=denom>0),0,1)
    return np.hypot(np.asarray(px)-ax-t*dx,np.asarray(py)-ay-t*dy),t

def bearing_deg(ax,ay,bx,by):
    return np.degrees(np.arctan2(np.asarray(bx)-ax,np.asarray(by)-ay))%360

def angle_diff_deg(a,b):
    return np.abs((np.asarray(a)-b+180)%360-180)
