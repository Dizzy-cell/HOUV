import open3d as o3d
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import torch

def get_pts(pcd):
    points = np.asarray(pcd.points)
    X = []
    Y = []
    Z = []
    for pt in range(points.shape[0]):
        X.append(points[pt][0])
        Y.append(points[pt][1])
        Z.append(points[pt][2])
    return np.asarray(X), np.asarray(Y), np.asarray(Z)


def set_axes_equal(ax):
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)
    plot_radius = 0.5*max([x_range, y_range, z_range])
    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def plot_single_pcd(points, save_path, rotation_matrix = None):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect('equal')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    if rotation_matrix:
        #rotation_matrix = np.asarray([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        pcd = pcd.transform(rotation_matrix)

    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('on')
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()

def plot_combine_pcd(points, points_vis, save_path):
    fig = plt.figure()
    
    ax = fig.add_subplot(211, projection='3d')
    ax.set_aspect('equal')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')

    points = points_vis
    ax = fig.add_subplot(212, projection='3d')
    ax.set_aspect('equal')
    pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    X, Y, Z = get_pts(pcd)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    set_axes_equal(ax)
    plt.axis('off')

    plt.savefig(save_path, format='png', dpi=600)
    plt.close()


def plot_grid_pcd(points_list, shape = [2,6], save_path = 'visulization', show_axis = 'off', dpi = 1200):
    fig = plt.figure()  
    num = shape[0] * shape[1]

    for i in range(shape[0]):
        for j in range(shape[1]):
            num = i * shape[1] + j
            points = points_list[num]
            ax = fig.add_subplot(shape[0],shape[1],num+1, projection='3d')
            ax.set_aspect('equal')
            pcd = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
            X, Y, Z = get_pts(pcd)
            t = Z
            ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
            ax.grid(False)
            ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
            set_axes_equal(ax)
            plt.axis(show_axis)


    plt.savefig(save_path, format='png', dpi=dpi)
    plt.close()

def visulation_max_loss(points, gt, predicts, loss, loss_max, save_path = 'Hard_predict'):
    idx = np.argmax(loss)
    a = loss[idx]
    if a > loss_max:
        loss_max = a
        plot_grid_pcd([points[idx].transpose(), predicts[idx], gt[idx]], shape=(1,3), save_path = save_path, show_axis='on')
        print("Loss max:{}".format(loss_max))

    return loss_max

def visulation_predict(points, gt, predicts, save_path = 'Hard_predict'):
    for idx in range(points.shape[0]):
        list_ap = []
        list_ap.append(points[idx].transpose())
        list_ap.append(predicts[idx])
        list_ap.append(gt[idx])
        
        plot_grid_pcd(list_ap, shape=(1, 3), save_path = save_path + str(idx) + '.png', show_axis='off', dpi = 1200)
    return