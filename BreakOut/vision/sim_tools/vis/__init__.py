from ..common import *


def plot_connector_3d(ax, imgw, imgh, conns, col_start=0, row_start=0,
                      col_step=1, row_step=1):

    pop_e = np.array([[x, -y, 0] for y in range(imgh) for x in range(imgw)])
    pop_i = np.array([[x, -y, 1] for y in range(imgh) for x in range(imgw)])
    pop_d = np.array([[x, -y, 10] for y in range(col_start, imgh, col_step) \
                                  for x in range(row_start, imgw, row_step)])

    # print(pop_e)
    ax.scatter(pop_e[:, 0], pop_e[:, 1], pop_e[:, 2], c='g')
    ax.scatter(pop_i[:, 0], pop_i[:, 1], pop_i[:, 2], c='r')
    ax.scatter(pop_d[:, 0], pop_d[:, 1], pop_d[:, 2], c='c')
    
    for i in range(len(conns[0])):
        src, dst, w, d = conns[0][i]
        w = np.clip(w*2, 0., 1.)
        xs = [pop_e[src, 0], pop_d[dst, 0]]
        ys = [pop_e[src, 1], pop_d[dst, 1]]
        zs = [pop_e[src, 2], pop_d[dst, 2]]
        ax.plot(xs, ys, zs, c=(0., 1., 0., w))

    for i in range(len(conns[1])):
        src, dst, w, d = conns[1][i]
        w = np.clip((-w)*2, 0., 1.)
        xs = [pop_i[src, 0], pop_d[dst, 0]]
        ys = [pop_i[src, 1], pop_d[dst, 1]]
        zs = [pop_i[src, 2], pop_d[dst, 2]]
        ax.plot(xs, ys, zs, c=(1., 0., 0., w))
        
    ax.set_xlabel('Cols (x)')
    ax.set_ylabel('Rows (y)')
    ax.set_zlabel('Z Label')

