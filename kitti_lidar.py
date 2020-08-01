import os


class KITTIObject:
    def __init__(self, root_dir, dataset_type, ):
        self.root_dir = root_dir
        self.dataset_type = dataset_type

        self.dataset_dir = os.path.join(self.root_dir, self.dataset_type)

        self.num_samples = len(os.listdir(self.dataset_dir))

        self.image_dir = os.path.join(self.dataset_dir, "image_2")
        self.label_dir = os.path.join(self.dataset_dir, "label_2")
        self.calib_dir = os.path.join(self.dataset_dir, "calib")

        self.depthpc_dir = os.path.join(self.dataset_dir, "depth_pc")
        self.lidar_dir = os.path.join(self.dataset_dir, "velodyne")
        self.depth_dir = os.path.join(self.dataset_dir, "depth")
        self.pred_dir = os.path.join(self.dataset_dir, "pred")

    def get_label_objects(self, idx):
        if idx > self.num_samples:
            return None
        label_filename = os.path.join(self.label_dir, "%06d.txt" % idx)
        # return utils.read_label(label_filename)