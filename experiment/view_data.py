import json
import numpy as np
from pathlib import Path
import open3d as o3d

COLOR20 = np.array(
    [[230, 25, 75], [60, 180, 75], [255, 225, 25], [0, 130, 200], [245, 130, 48],
     [145, 30, 180], [70, 240, 240], [240, 50, 230], [210, 245, 60], [250, 190, 190],
     [0, 128, 128], [230, 190, 255], [170, 110, 40], [255, 250, 200], [128, 0, 0],
     [170, 255, 195], [128, 128, 0], [255, 215, 180], [0, 0, 128], [128, 128, 128]])

def build_pointcloud(pc, cid: np.ndarray = None):
    assert pc.shape[1] == 3 and len(pc.shape) == 2, f"Point cloud is of size {pc.shape} and cannot be displayed!"
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pc)
    if cid is not None:
        assert cid.shape[0] == pc.shape[0], f"Point and color id must have same size {cid.shape[0]}, {pc.shape[0]}"
        assert cid.ndim == 1, f"color id must be of size (N,) currently ndim = {cid.ndim}"
        point_cloud.colors = o3d.utility.Vector3dVector(COLOR20[cid % COLOR20.shape[0]] / 255.)

    return point_cloud

if __name__ == "__main__":
    base_folder = Path('/localhome/yma50/Development/multibody-sync/dataset/mbs-shapepart')
    with (base_folder / "meta.json").open() as f:
        meta = json.load(f)
    split = 'train'
    data_ids = meta[split]

    idx = 0

    data_path = base_folder / "data" / ("%06d.npz" % data_ids[idx])
    datum = np.load(data_path, allow_pickle=True)

    raw_pc = datum['pc'].astype(np.float32)
    raw_segm = datum['segm']
    raw_trans = datum['trans'].item()

    segmented_pcds = []
    for view_i in range(raw_pc.shape[0]):
        segmented_pcds.append(build_pointcloud(raw_pc[view_i], raw_segm[view_i]))
        segmented_pcds[-1].translate([view_i, 0.0, 0.0])
    o3d.visualization.draw_geometries(segmented_pcds)

    with open('./trans.json', 'w+') as f:
        d = {}
        for k, v in raw_trans.items():
            if isinstance(v, np.ndarray):
                d[k] = v.tolist()
            else:
                d[k] = [v0.tolist() for v0 in v]
        json.dump(d, f)