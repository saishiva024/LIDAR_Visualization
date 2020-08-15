import numpy as np
import cv2


class Object3D:
    def __init__(self, label):
        data = label.split(' ')

        self.classification = data[0]
        self.truncation = float(data[1])
        self.occlusion = int(data[2])
        self.alpha_observation_angle = float(data[3])

        self.xmin_left = float(data[4])
        self.ymin_top = float(data[5])
        self.xmax_right = float(data[6])
        self.ymax_bottom = float(data[7])

        self.bbox2d = np.array([self.xmin_left, self.ymin_top, self.xmax_right, self.ymax_bottom])

        self.height = float(data[8])
        self.width = float(data[9])
        self.length = float(data[10])

        self.obj3d_location_camera_coords = (float(data[11]), float(data[12]), float(data[13]))

        self.yaw_rotation_y = float(data[14])


class Calibration:
    def __init__(self, calib_filename):
        calibration = self.read_calibration_file(calib_filename)
        self.P = calibration["P2"]
        self.P = np.reshape(self.P, [3, 4])
        self.V2C = calibration["Tr_velo_to_cam"]
        self.V2C = np.reshape(self.V2C, [3, 4])
        self.C2V = inverse_rigid_transform(self.V2C)
        self.R0 = calibration["R0_rect"]
        self.R0 = np.reshape(self.R0, [3, 3])

        self.c_u = self.P[0, 2]
        self.c_v = self.P[1, 2]
        self.f_u = self.P[0, 0]
        self.f_v = self.P[1, 1]
        self.b_x = self.P[0, 3] / (-self.f_u)
        self.b_y = self.P[1, 3] / (-self.f_v)

    def read_calibration_file(self, filename):
        data = {}
        with open(filename, "r") as file:
            for line in file.readlines():
                line = line.rstrip()
                if len(line) == 0:
                    continue
                key, value = line.split(":", 1)
                try:
                    data[key] = np.array([float(x) for x in value.split()])
                except ValueError:
                    pass
        return data

    def project_lidar_to_image(self, points3d_lidar):
        points3d_rect = self.project_lidar_to_rect(points3d_lidar)
        return self.project_rect_to_image(points3d_rect)

    def project_lidar_to_rect(self, points3d_lidar):
        points3d_ref = self.project_lidar_to_ref(points3d_lidar)
        return self.project_ref_to_rect(points3d_ref)

    def project_lidar_to_ref(self, points3d_lidar):
        points3d_lidar = self.cart2hom(points3d_lidar)
        return np.dot(points3d_lidar, np.transpose(self.V2C))

    def project_ref_to_rect(self, points3d_ref):
        return np.transpose(np.dot(self.R0, np.transpose(points3d_ref)))

    def cart2hom(self, points3d):
        """In: nx3 - Points in Cartesian
           Out: nx4 - Points in Homogeneous
        """
        num_points = points3d.shape[0]
        points3d_hom = np.hstack((points3d, np.ones((num_points, 1))))
        return points3d_hom

    def project_rect_to_image(self, points3d_rect):
        points3d_rect = self.cart2hom(points3d_rect)
        points2d = np.dot(points3d_rect, np.transpose(self.P))
        points2d[:, 0] /= points2d[:, 2]
        points2d[:, 1] /= points2d[:, 2]
        return points2d[:, 0:2]

    def project_rect_to_lidar(self, points3d_rect):
        points3d_ref = self.project_rect_to_ref(points3d_rect)
        return self.project_ref_to_lidar(points3d_ref)

    def project_rect_to_ref(self, points3d_rect):
        return np.transpose(np.dot(np.linalg.inv(self.R0), np.transpose(points3d_rect)))

    def project_ref_to_lidar(self, points3d_ref):
        points3d_ref = self.cart2hom(points3d_ref)
        return np.dot(points3d_ref, np.transpose(self.C2V))


def inverse_rigid_transform(tr):
    inv_tr = np.zeros_like(tr)
    inv_tr[0:3, 0:3] = np.transpose(tr[0:3, 0:3])
    inv_tr[0:3, 3] = np.dot(-np.transpose(tr[0:3, 0:3]), tr[0:3, 3])
    return inv_tr


def load_image(image_filename):
    return cv2.imread(image_filename)


def read_label(label_file_name):
    objects = []
    with open(label_file_name, 'r') as lf:
        for line in lf.readlines():
            objects.append(Object3D(line.rstrip()))
    return objects


def rotate_y(p):
    cp = np.cos(p)
    sp = np.sin(p)

    return np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])


def project_3d_to_image2d(points3d, P):
    n = points3d.shape[0]
    points3d_extended = np.hstack((points3d, np.ones((n, 1))))
    points2d = np.dot(points3d_extended, np.transpose(P))
    points2d[:, 0] /= points2d[:, 2]
    points2d[:, 1] /= points2d[:, 2]
    return points2d[:, 0:2]


def compute_bbox3d(obj, P):
    R = rotate_y(obj.yaw_rotation_y)

    l = obj.length
    w = obj.width
    h = obj.height

    # 3D BBox Corners
    x_corners = [l / 2, l / 2, -l / 2, -l / 2, l / 2, l / 2, -l / 2, -l / 2]
    y_corners = [0, 0, 0, 0, -h, -h, -h, -h]
    z_corners = [w / 2, -w / 2, -w / 2, w / 2, w / 2, -w / 2, -w / 2, w / 2]

    # rotate and translate 3D BBox
    corners3d = np.dot(R, np.vstack([x_corners, y_corners, z_corners]))

    corners3d[0, :] = corners3d[0, :] + obj.obj3d_location_camera_coords[0]
    corners3d[1, :] = corners3d[1, :] + obj.obj3d_location_camera_coords[1]
    corners3d[2, :] = corners3d[2, :] + obj.obj3d_location_camera_coords[2]

    if np.any(corners3d[2, :] < 0.1): # Draw 3D BBoxes for objects which are infront of Camera
        corners2d = None
        return corners2d, np.transpose(corners3d)

    # Project 3D BBox onto Image plane
    corners2d = project_3d_to_image2d(np.transpose(corners3d), P)
    return corners2d, np.transpose(corners3d)


def get_velodyne_scan_points(lidar_filename, n_vec=4):
    lidar_scanpoints = np.fromfile(lidar_filename, np.float32)
    lidar_scanpoints = lidar_scanpoints.reshape((-1, n_vec))
    return lidar_scanpoints
