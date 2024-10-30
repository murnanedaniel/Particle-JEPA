from trackml.dataset import load_event
import numpy as np

def prep_hits(file):
    hits, cells, particles, truth = load_event(file)
    hits["r"] = np.sqrt(hits["x"]**2 + hits["y"]**2)
    theta = np.arctan2(hits["r"], hits["z"])
    hits["eta"] = -np.log(np.tan(theta / 2.0))
    hits["phi"] = np.arctan2(hits["y"], hits["x"])
    hits = hits.merge(truth, on="hit_id", how="left")
    hits = hits.merge(particles, on="particle_id", how="left")
    hits["pt"] = hits["px"]**2 + hits["py"]**2
    return hits, particles