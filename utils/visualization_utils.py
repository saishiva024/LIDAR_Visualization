import numpy as np
import mayavi.mlab as mlab


def draw_groundtruth_3dbboxes(groundtruth_3dbboxes, fig, color=(1, 1, 1), line_width=1, draw_text=True,
                              text_scale=(1, 1, 1), color_list=None, label=""):
    num_bboxes = len(groundtruth_3dbboxes)
    for idx in range(num_bboxes):
        bbox = groundtruth_3dbboxes[idx]
        if color_list is not None:
            color = color_list[idx]
        if draw_text:
            mlab.text3d(bbox[4, 0], bbox[4, 1], bbox[4, 2], label, scale=text_scale, color=color, figure=fig)
        for k in range(0, 4):
            i, j = k, (k + 1) % 4
            mlab.plot3d([bbox[i, 0], bbox[j, 0]],
                        [bbox[i, 1], bbox[j, 1]],
                        [bbox[i, 2], bbox[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)
            i, j = k + 4, (k + 1) % 4 + 4
            mlab.plot3d([bbox[i, 0], bbox[j, 0]],
                        [bbox[i, 1], bbox[j, 1]],
                        [bbox[i, 2], bbox[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)
            i, j = k, k + 4
            mlab.plot3d([bbox[i, 0], bbox[j, 0]],
                        [bbox[i, 1], bbox[j, 1]],
                        [bbox[i, 2], bbox[j, 2]],
                        color=color,
                        tube_radius=None,
                        line_width=line_width,
                        figure=fig)
    return fig


def draw_lidar_simple(pointcloud, color=None):
    figure = mlab.figure(figure=None, bgcolor=(0, 0, 0), fgcolor=None, engine=None, size=(1500, 1000))
    if color is None:
        color = pointcloud[:, 2]
    mlab.points3d(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                  color, color=None, mode="point", colormap="gnuplot", scale_factor=1, figure=figure)  # points
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)  # origin
    axes = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0), tube_radius=None, figure=figure)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0), tube_radius=None, figure=figure)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1), tube_radius=None, figure=figure)
    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991],
              distance=62.0, figure=figure)
    return figure


def draw_lidar(pointcloud, color=None, fig=None, bg_color=(0, 0, 0), points_scale=0.3, points_mode="sphere",
               points_color=None, color_by_intensity=False, pointcloud_label=False):
    points_mode = "point"
    if fig is None:
        fig = mlab.figure(figure=None, bgcolor=bg_color, fgcolor=None, engine=None, size=(1500, 1000))
    if color is None:
        color = pointcloud[:, 2]
    if pointcloud_label:
        color = pointcloud[:, 4]
    if color_by_intensity:
        color = pointcloud[:, 2]
    mlab.points3d(pointcloud[:, 0], pointcloud[:, 1], pointcloud[:, 2],
                  color, color=points_color, mode=points_mode, colormap="gnuplot",
                  scale_factor=points_scale, figure=fig)  # points
    mlab.points3d(0, 0, 0, color=(1, 1, 1), mode="sphere", scale_factor=0.2)  # origin
    axes = np.array([[2.0, 0.0, 0.0, 0.0], [0.0, 2.0, 0.0, 0.0], [0.0, 0.0, 2.0, 0.0]], dtype=np.float64)
    mlab.plot3d([0, axes[0, 0]], [0, axes[0, 1]], [0, axes[0, 2]],
                color=(1, 0, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[1, 0]], [0, axes[1, 1]], [0, axes[1, 2]],
                color=(0, 1, 0), tube_radius=None, figure=fig)
    mlab.plot3d([0, axes[2, 0]], [0, axes[2, 1]], [0, axes[2, 2]],
                color=(0, 0, 1), tube_radius=None, figure=fig)

    fov = np.array([[20.0, 20.0, 0.0, 0.0], [20.0, -20.0, 0.0, 0.0]], dtype=np.float64)
    mlab.plot3d([0, fov[0, 0]], [0, fov[0, 1]], [0, fov[0, 2]],
                color=(1, 1, 1), tube_radius=None, line_width=1, figure=fig)
    mlab.plot3d([0, fov[1, 0]], [0, fov[1, 1]], [0, fov[1, 2]],
                color=(1, 1, 1), tube_radius=None, line_width=1, figure=fig)

    x1, y1 = 0, -20
    x2, y2 = 40, 20

    mlab.plot3d([x1, x1], [y1, y2], [0, 0],
                color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x2, x2], [y1, y2], [0, 0],
                color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y1, y1], [0, 0],
                color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)
    mlab.plot3d([x1, x2], [y2, y2], [0, 0],
                color=(0.5, 0.5, 0.5), tube_radius=0.1, line_width=1, figure=fig)

    mlab.view(azimuth=180, elevation=70, focalpoint=[12.0909996, -1.04700089, -2.03249991],
              distance=62.0, figure=fig)
    return fig
