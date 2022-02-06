import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
import cv2
import pickle


source_position_dict = {'a': (13, 6), 'b': (27, 3), 'c': (21, 4), 'd': (3, 3), 'e': (14, 3)}


def raw_data_to_grid(coordinate_x, coordinate_y, concentration, velocity_x, velocity_y, word_size=15, grid_resolution=30):
    x, y = np.mgrid[0:word_size:grid_resolution * 1j, 0:word_size:grid_resolution * 1j]
    concentration_grid = griddata((coordinate_x, coordinate_y), concentration, (x, y), method="cubic", fill_value=0)
    velocity_x_grid = griddata((coordinate_x, coordinate_y), velocity_x, (x, y), method="cubic", fill_value=0)
    velocity_y_grid = griddata((coordinate_x, coordinate_y), velocity_y, (x, y), method="cubic", fill_value=0)
    return concentration_grid, velocity_x_grid, velocity_y_grid


def raw_image_to_map(map_image, grid_resolution=30, dilate=True):
    if len(map_image.shape) > 2:
        map_image = cv2.cvtColor(map_image, cv2.COLOR_BGR2GRAY)
    ix, iy = map_image.shape
    edge_size = (max(ix, iy) - min(ix, iy)) // 2
    if ix > iy:
        map_image = map_image[edge_size:edge_size + min(ix, iy), :]
    else:
        map_image = map_image[:, edge_size:edge_size + min(ix, iy)]
    map_image = cv2.resize(map_image, (grid_resolution, grid_resolution))
    _, map_image = cv2.threshold(map_image, 125, 255, cv2.THRESH_BINARY_INV)

    # 膨胀
    if dilate:
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        map_image = cv2.dilate(map_image, kernel=kernel)

    map_image = map_image.astype(np.float64)
    map_image /= 255.0
    return map_image


def plot_grid(map_grid, concentration_grid, velocity_x_grid, velocity_y_grid, word_size=15, grid_resolution=30):
    plt.figure(figsize=(16, 5))
    ax = plt.subplot(131)
    plt.imshow(map_grid, cmap='gray')
    plt.title('map')
    ax = plt.subplot(132)
    plt.imshow(concentration_grid, cmap='jet')
    plt.colorbar()
    plt.title('concentration')
    ax = plt.subplot(133)
    x, y = np.mgrid[0:word_size:grid_resolution * 1j, 0:word_size:grid_resolution * 1j]
    plt.quiver(y, x, velocity_y_grid, velocity_x_grid, color='C0', angles='xy')
    ax.invert_yaxis()
    plt.axis('equal')
    plt.title('velocity')
    plt.show()


def plot_grid2(map_prefix, map_grid, concentration_grid, velocity_x_grid, velocity_y_grid, word_size=15, grid_resolution=30):
    plt.figure(0, figsize=(5, 4), dpi=300)
    plt.clf()
    plt.imshow(concentration_grid)
    # x, y = np.mgrid[0:grid_resolution:grid_resolution * 1j, 0:grid_resolution:grid_resolution * 1j]
    # plt.quiver(y, x, velocity_y_grid, velocity_x_grid, color='C0', angles='xy')
    # plt.gca().invert_yaxis()
    for i in range(map_grid.shape[0]):
        for j in range(map_grid.shape[1]):
            if map_grid[i, j] > 0:
                plt.gca().add_patch(
                    plt.Rectangle((j-0.5, i-0.5), 1.0, 1.0,
                                  facecolor="white"))
    plt.colorbar()
    plt.xlabel('y')
    plt.ylabel('x')
    # plt.gca().set_aspect('equal', adjustable='box')
    # plt.show()
    plt.savefig(f'output_image/{map_prefix}.png', format='png')


def read_data_from_file(map_prefix: str, dilate=True):
    m = raw_image_to_map(cv2.imread(f'raw_data/{map_prefix}.png'), dilate=dilate)
    d = np.loadtxt(f'raw_data/{map_prefix}.csv', np.float64, delimiter=",", skiprows=1)
    c, vx, vy = raw_data_to_grid(d[:, 1], d[:, 2], d[:, 6], d[:, 4], d[:, 5])

    # 去除障碍物处的数据
    c[m > 0.5] = 0.0
    vx[m > 0.5] = 0.0
    vy[m > 0.5] = 0.0

    # 去除小于0的浓度数据
    c[c < 0] = 0.0

    return m, c, vx, vy


def get_gas_source_position(map_prefix: str):
    return source_position_dict[map_prefix]


def load_processed_data():
    with open('processed_data.data', 'rb') as fp:
        processed_data = pickle.load(fp)
    return processed_data


if __name__ == '__main__':
    processed_data = dict()

    processed_data['a'] = dict()
    map_grid, concentration_grid, velocity_x_grid, velocity_y_grid = read_data_from_file('a')
    processed_data['a']['map_grid'] = map_grid
    processed_data['a']['concentration_grid'] = concentration_grid
    processed_data['a']['velocity_x_grid'] = velocity_x_grid
    processed_data['a']['velocity_y_grid'] = velocity_y_grid
    processed_data['a']['source_position'] = source_position_dict['a']

    plot_grid(map_grid, concentration_grid, velocity_x_grid, velocity_y_grid)

    processed_data['b'] = dict()
    map_grid, concentration_grid, velocity_x_grid, velocity_y_grid = read_data_from_file('b')
    processed_data['b']['map_grid'] = map_grid
    processed_data['b']['concentration_grid'] = concentration_grid
    processed_data['b']['velocity_x_grid'] = velocity_x_grid
    processed_data['b']['velocity_y_grid'] = velocity_y_grid
    processed_data['b']['source_position'] = source_position_dict['b']

    plot_grid(map_grid, concentration_grid, velocity_x_grid, velocity_y_grid)

    processed_data['c'] = dict()
    map_grid, concentration_grid, velocity_x_grid, velocity_y_grid = read_data_from_file('c')
    processed_data['c']['map_grid'] = map_grid
    processed_data['c']['concentration_grid'] = concentration_grid
    processed_data['c']['velocity_x_grid'] = velocity_x_grid
    processed_data['c']['velocity_y_grid'] = velocity_y_grid
    processed_data['c']['source_position'] = source_position_dict['c']

    plot_grid(map_grid, concentration_grid, velocity_x_grid, velocity_y_grid)

    processed_data['d'] = dict()
    map_grid, concentration_grid, velocity_x_grid, velocity_y_grid = read_data_from_file('d')
    processed_data['d']['map_grid'] = map_grid
    processed_data['d']['concentration_grid'] = concentration_grid
    processed_data['d']['velocity_x_grid'] = velocity_x_grid
    processed_data['d']['velocity_y_grid'] = velocity_y_grid
    processed_data['d']['source_position'] = source_position_dict['d']

    plot_grid(map_grid, concentration_grid, velocity_x_grid, velocity_y_grid)

    processed_data['e'] = dict()
    map_grid, concentration_grid, velocity_x_grid, velocity_y_grid = read_data_from_file('e')
    processed_data['e']['map_grid'] = map_grid
    processed_data['e']['concentration_grid'] = concentration_grid
    processed_data['e']['velocity_x_grid'] = velocity_x_grid
    processed_data['e']['velocity_y_grid'] = velocity_y_grid
    processed_data['e']['source_position'] = source_position_dict['e']

    plot_grid(map_grid, concentration_grid, velocity_x_grid, velocity_y_grid)

    with open('processed_data.data', 'wb') as fp:
        pickle.dump(processed_data, fp)

    print('successful')


    # for map_prefix in ['a', 'b', 'c', 'd', 'e']:
    #     map_grid, concentration_grid, velocity_x_grid, velocity_y_grid = read_data_from_file(map_prefix, False)
    #     plot_grid2(map_prefix, map_grid, concentration_grid, velocity_x_grid, velocity_y_grid)

