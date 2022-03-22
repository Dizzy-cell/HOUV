import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import open3d as o3d


def plot_pcd(ax, pcd, color=None, cmap='viridis', size=4, alpha=0.9, azim=60, elev=0):
	if color is None:
		color = pcd[:, 0]
		vmin = -2
		vmax = 1.5
	else:
		vmin = 0
		vmax = 1
	ax.view_init(azim=azim, elev=elev)
	ax.scatter(pcd[:, 0], pcd[:, 1], pcd[:, 2], c=color, s=size, cmap=cmap, vmin=vmin, vmax=vmax, alpha=alpha)
	lims = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
	min_lim = min(pcd.min() * 0.9, lims.min())
	max_lim = max(pcd.max() * 0.9, lims.max())
	for axis in 'xyz':
		getattr(ax, 'set_{}lim'.format(axis))((min_lim, max_lim))
	ax.set_axis_off()


def plot_matches(ax, mpts1, mpts2, color=None, cmap='viridis', size=4, alpha=0.9, azim=60, elev=0):
	if color is None:
		color = np.arange(mpts1.shape[0]) / (mpts1.shape[0] - 1)
	if cmap is not None:
		cmap = cm.get_cmap(cmap)
		color = cmap(color)

	ax.view_init(azim=azim, elev=elev)

	for k in range(mpts1.shape[0]):
		ptp = np.array([mpts1[k], mpts2[k]])
		ax.plot(ptp[:, 0], ptp[:, 1], ptp[:, 2], color=color[k], marker='o', markersize=12)


def plot_gmm(ax, mix, mu, cov, color=None, cmap='viridis', azim=60, elev=0, numWires=15, wireframe=True):
	if color is None:
		color = np.arange(mix.shape[0]) / (mix.shape[0] - 1)
	if cmap is not None:
		cmap = cm.get_cmap(cmap)
		color = cmap(color)

	u = np.linspace(0.0, 2.0 * np.pi, numWires)
	v = np.linspace(0.0, np.pi, numWires)
	X = np.outer(np.cos(u), np.sin(v))
	Y = np.outer(np.sin(u), np.sin(v))
	Z = np.outer(np.ones_like(u), np.cos(v)) 
	XYZ = np.stack([X.flatten(), Y.flatten(), Z.flatten()])

	alpha = mix / mix.max()
	ax.view_init(azim=azim, elev=elev)

	for k in range(mix.shape[0]):
		# find the rotation matrix and radii of the axes
		U, s, V = np.linalg.svd(cov[k])
		x, y, z = V.T @ (np.sqrt(s)[:, None] * XYZ) + mu[k][:, None]
		x = x.reshape(numWires, numWires)
		y = y.reshape(numWires, numWires)
		z = z.reshape(numWires, numWires)
		if wireframe:
			ax.plot_wireframe(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])
		else:
			ax.plot_surface(x, y, z, rstride=1, cstride=1, color=color[k], alpha=alpha[k])


def visualize(inputs):
	for i in range(len(inputs)):
		inputs[i] = inputs[i].detach().cpu().numpy()
	p1, gamma1, pi1, mu1, sigma1, p2, gamma2, pi2, mu2, sigma2, \
		p1_trans, init_r_err, init_t_err, init_rmse, r_err, t_err, rmse = inputs

	fig = plt.figure(figsize=(8, 8))
	title = 'Rotation error {:.2f}\nTranslation error {:.4f}\nRMSE {:.4f}'

	ax = fig.add_subplot(221, projection='3d')
	plot_pcd(ax, p1, cmap='Reds')
	plot_pcd(ax, p2, cmap='Blues')
	ax.set_title(title.format(init_r_err, init_t_err, init_rmse))

	ax = fig.add_subplot(222, projection='3d')
	plot_pcd(ax, p1_trans, cmap='Reds')
	plot_pcd(ax, p2, cmap='Blues')
	ax.set_title(title.format(r_err, t_err, rmse))

	ax = fig.add_subplot(223, projection='3d')
	color1 = np.argmax(gamma1, axis=1) / (gamma1.shape[1] - 1)
	plot_pcd(ax, p1, color1)
	plot_gmm(ax, pi1, mu1, sigma1)
	ax.set_title('Source GMM')

	ax = fig.add_subplot(224, projection='3d')
	color2 = np.argmax(gamma2, axis=1) / (gamma2.shape[1] - 1)
	plot_pcd(ax, p2, color2)
	plot_gmm(ax, pi2, mu2, sigma2)
	ax.set_title('Target GMM')

	plt.tight_layout()
	return fig


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

def plot_grid_pcd(points_list, shape = [2,6], save_path = 'visulization', title = None):

    plt.subplots_adjust(top=1,bottom=0,left=0,right=1,hspace=0,wspace=0)
    plt.margins(0,0)

    fig = plt.figure() 
    fig.tight_layout()

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
            plt.axis('off')

    plt.title('R_err:{}'.format(title))
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()


def aligned_plot_grid_pcd(points_a, points_b, a, b, save_path = 'aligned_visulization.png', title = None):
    # from IPython import embed
    # embed()


    fig = plt.figure()  
    points = points_a
    ax = fig.add_subplot(111, projection='3d')
    pcda = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    X, Y, Z = get_pts(pcda)
    pcda = np.asarray(pcda.points)
    t = Z
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    # ax.grid(False)
    # ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # set_axes_equal(ax)

    points = points_b
    pcdb = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    X, Y, Z = get_pts(pcdb)
    pcdb = np.asarray(pcdb.points)
    t = Z
    X = [i+4 for i in X]
    ax.scatter(X, Y, Z, c=t, cmap='jet', marker='o', s=0.5, linewidths=0)
    ax.grid(False)
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 1.0))
    # set_axes_equal(ax)
    

    for u, v in enumerate(zip(a,b)):
        i, j = v
        X = [pcda[i][0], pcdb[j][0]+4]
        Y = [pcda[i][1], pcdb[j][1]]
        Z = [pcda[i][2], pcdb[j][2]]
        ax.plot(X,Y,Z, c = 'r', linestyle='-', alpha=0.5)
        if u > 20:
            break

    plt.axis('off')
    plt.title('Aligned exmaple'.format(title))
    plt.savefig(save_path, format='png', dpi=600)
    plt.close()



def analyseDis(dis, bin = 25,savename = 'ICP.png'):
    dis = np.sqrt((dis * dis).sum(axis = 1))
    step = 0.25 / bin
    dis = dis / step
    dis[dis > bin] = bin

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter

    plt.xlabel('Bins')
    plt.ylabel('Probability')

    fig= plt.figure(figsize=(8, 4),dpi=100)
    n, bins, patches = plt.hist(dis, 50, density=True, facecolor='g', alpha=0.5, label = 'Contrained')
    n, bins, patches = plt.hist(dis, 50, density=True, facecolor='r', alpha=0.5, label = 'Sigmoid')
    n, bins, patches = plt.hist(dis, 50, density=True, facecolor='b', alpha=0.5, label = 'Sine')

    plt.legend()
    plt.legend(loc='upper left')
    plt.savefig(savename)
    
    return 




def getDis(filename):
    import h5py
    f = h5py.File(filename, 'r')
    data = np.array(f['results'][:].astype('float32'))
    f.close()
    return data[:,:3,3]

def analyseDises(dises, bin = 25,savename = 'DIS.png'):

    for i, dis in enumerate(dises):
        dis = np.sqrt((dis * dis).sum(axis = 1))
        step = 0.25 / bin
        dis = dis / step
        dis[dis > bin] = bin
        dises[i] = dis

    import matplotlib.pyplot as plt
    import matplotlib as mpl
    from matplotlib.ticker import FuncFormatter

    fig= plt.figure(figsize=(8, 4),dpi=100)

    plt.xlabel('Bins')
    plt.ylabel('Probability')
    n, bins, patches = plt.hist(dises[0], 50, density=True, facecolor='g', alpha=0.5, label = 'Uncontrained')
    n, bins, patches = plt.hist(dises[1], 50, density=True, facecolor='r', alpha=0.5, label = 'Sigmoid')
    n, bins, patches = plt.hist(dises[2], 50, density=True, facecolor='b', alpha=0.5, label = 'Sine')

    plt.legend()
    #plt.legend(loc='upper left')
    plt.savefig(savename)
    
    # from IPython import embed
    # embed()
    return 



