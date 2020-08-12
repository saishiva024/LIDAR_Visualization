import os
from utils.kitti_utils import read_label, get_velodyne_scan_points, load_image, compute_bbox3d, Calibration
from utils.visualization_utils import draw_lidar_simple, draw_lidar, draw_groundtruth_3dbboxes
import mayavi.mlab as mlab
import numpy as np
import matplotlib.pyplot as plt
import cv2


class KITTIObject:
    def __init__(self, root_dir, dataset_type, ):
        self.root_dir = root_dir
        self.dataset_type = dataset_type

        self.dataset_dir = os.path.join(self.root_dir, self.dataset_type)

        self.num_samples = len(os.listdir(self.dataset_dir))

        self.image_dir = os.path.join(self.dataset_dir, "image_2")
        self.label_dir = os.path.join(self.dataset_dir, "label_2")
        self.calib_dir = os.path.join(self.dataset_dir, "calib")
        self.lidar_dir = os.path.join(self.dataset_dir, "velodyne")

    def get_label_objects(self, idx):
        if idx > self.num_samples:
            return None
        label_filename = os.path.join(self.label_dir, "%06d.txt" % idx)
        return read_label(label_filename)

    def get_lidar_data(self, idx, n_vec=4):
        lidar_filename = os.path.join(self.lidar_dir, "%06d.bin" % idx)
        return get_velodyne_scan_points(lidar_filename, n_vec)

    def get_calibration(self, idx):
        calib_filename = os.path.join(self.calib_dir, "%06d.txt" % idx)
        return Calibration(calib_filename)

    def get_image(self, idx):
        image_filename = os.path.join(self.image_dir, "%06d.png" % idx)
        return load_image(image_filename)


def get_lidar_index_in_image_fov(pcd, calib, xmin, ymin, xmax, ymax, clip_distance=2.0):
    points2d = calib.project_lidar_to_image(pcd)
    fov_idxs = ((points2d[:, 0] < xmax) & (points2d[:, 0] >= xmin) &
                (points2d[:, 1] < ymax) & (points2d[:, 1] >= ymin))
    fov_idxs = fov_idxs & (pcd[:, 0] > clip_distance)
    return fov_idxs


def show_lidar_data(point_cloud_data, objects, calibration, figure,
                    img_fov=False, img_width=None, img_height=None, cam_img=None, pc_label=False, save=False):
    print(("All point num: ", point_cloud_data.shape[0]))
    if img_fov:
        pcd_index = get_lidar_index_in_image_fov(point_cloud_data[:, :3], calibration, 0, 0, img_width, img_height)
        point_cloud_data = point_cloud_data[pcd_index, :]
        print(("FOV point num: ", point_cloud_data.shape))
    draw_lidar(point_cloud_data, fig=figure, pointcloud_label=pc_label)

    color = (0, 1, 0)

    for obj in objects:
        if obj.classification == "DontCare":
            continue
        _, bbox3d = compute_bbox3d(obj, calibration.P)
        bbox3d_velodyne = calibration.project_rect_to_lidar(bbox3d)

        draw_groundtruth_3dbboxes([bbox3d_velodyne], fig=figure, color=color, label=obj.classification)
    mlab.show(1)


def get_lidar_data_in_image_fov(pcd, calib, xmin, ymin, xmax, ymax, clip_distance=2.0):
    points2d = calib.project_lidar_to_image(pcd)
    fov_idxs = ((points2d[:, 0] < xmax) & (points2d[:, 0] >= xmin) &
                (points2d[:, 1] < ymax) & (points2d[:, 1] >= ymin))
    fov_idxs = fov_idxs & (pcd[:, 0] > clip_distance)
    imgfov_pcd = pcd[fov_idxs, :]

    return imgfov_pcd, points2d, fov_idxs


def show_lidar_data_on_image(point_cloud_data, img, calib, img_width, img_height):
    img = np.copy(img)
    imgfov_pcd, points2d, fov_indices = get_lidar_data_in_image_fov(point_cloud_data, calib, 0, 0, img_width, img_height)
    imgfov_points2d = points2d[fov_indices, :]
    imgfov_pc_rect = calib.project_lidar_to_rect(imgfov_pcd)

    cmap = plt.cm.get_cmap("hsv", 256)
    cmap = np.array([cmap(i) for i in range(256)])[:, :3] * 255

    for i in range(imgfov_points2d.shape[0]):
        depth = imgfov_pc_rect[i, 2]
        color = cmap[int(640.0 / depth), :]
        cv2.circle(img, (int(np.round(imgfov_points2d[i, 0])), int(np.round(imgfov_points2d[i, 1]))),
                   2, color=tuple(color), thickness=-1)
    return img

