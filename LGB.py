"""
Linde-Buzo-Gray / Generalized Lloyd algorithm implementation in Python 3.
"""
import numpy as np

LABEL_COLOR_MAP = {-1: 'white',
                   0: 'gray',
                   1: 'red',
                   2: 'blue',
                   3: 'green',
                   4: 'black',
                   5: 'yellow',
                   6: 'cyan',
                   7: 'magenta',
                   8: 'tab:purple',
                   }

def generate_codebook(data, size_codebook, epsilon=0.00001):
    """
    :param data: input data with N d-dimensional vectors
    :param size_codebook: codebook size
    :param epsilon: convergence value
    """
    _size_data = len(data)
    _dim = len(data[0])

    data = np.array(data)
    # calculate initial codevector: average vector of whole input data
    c0 = np.mean(data, axis=0)

    # calculate the average distortion
    avg_dist = np.mean(np.sum((data - c0) ** 2, axis=1))

    codebook = []
    codebook.append(c0)
    codebook_abs_weights = [_size_data]
    codebook_rel_weights = [1.0]
    codebook = np.array(codebook)

    # split codevectors until we have have enough
    while len(codebook) < size_codebook:
        codebook, labels, codebook_abs_weights, codebook_rel_weights, avg_dist = split_codebook(data, codebook, epsilon, avg_dist)

    return codebook, labels, codebook_abs_weights, codebook_rel_weights


def split_codebook(data, codebook, epsilon, initial_avg_dist):
    """
    Split the codebook so that each codevector in the codebook is split into two.
    :param data: input data
    :param codebook: input codebook. its codevectors will be split into two.
    :param epsilon: convergence value
    :param initial_avg_dist: initial average distortion
    :return Tuple with new codebook, codebook absolute weights and codebook relative weights
    """
    _size_data = len(data)
    _dim = len(data[0])

    # split codevectors
    codebook1 = codebook * (1.0 + epsilon)
    codebook2 = codebook * (1.0 - epsilon)
    codebook = np.vstack((codebook1, codebook2))

    len_codebook = len(codebook)
    abs_weights = [0] * len_codebook
    rel_weights = [0.0] * len_codebook

    # try to reach a convergence by minimizing the average distortion. this is done by moving the codevectors step by
    # step to the center of the points in their proximity
    avg_dist = 0
    err = epsilon + 1
    num_iter = 0
    while err > epsilon:
        # find closest codevectors for each vector in data (find the proximity of each codevector)
        closest_c_list = np.zeros((_size_data, _dim))    # list that contains the nearest codevector for each input data vector

        new_data = np.repeat(data[:, :, np.newaxis], len(codebook), axis=-1)
        new_dist = np.sum((new_data - codebook.T[np.newaxis, :, :])**2, axis=1)
        new_indexes = np.argmin(new_dist, axis=1)

        ids = np.arange(0, len(data))

        # update codebook: recalculate each codevector so that it sits in the center of the points in their proximity
        for i_c in range(len_codebook): # for each codevector index
            vecs = data[new_indexes==i_c]  # get its proximity input vectors

            new_c = np.mean(np.array(vecs), axis=0)    # calculate the new center
            codebook[i_c] = new_c                   # update in codebook

             # update in input vector index -> codevector mapping list
            closest_c_list[ids[new_indexes==i_c]] = new_c

            # update the weights
            abs_weights[i_c] = len(vecs)
            rel_weights[i_c] = len(vecs) / _size_data

        # recalculate average distortion value
        prev_avg_dist = avg_dist if avg_dist > 0 else initial_avg_dist
        closest_c_list = np.array(closest_c_list)

        avg_dist = np.mean(np.sum((data - closest_c_list)**2, axis=1))

        # recalculate the new error value
        err = (prev_avg_dist - avg_dist) / prev_avg_dist

        num_iter += 1

    return codebook, new_indexes, abs_weights, rel_weights, avg_dist




def lgb_simple_example():
    import time
    import random
    import matplotlib.pyplot as plt

    NUM_AREAS = SIZE_CODEBOOK = 8
    NUM_DIMS = 2
    NUM_POINTS_PER_AREA = 1000

    AREA_MIN_MAX = (-20, 20)

    random.seed(0)

    # create random centroids for NUM_AREAS areas
    area_centroids = [tuple([random.uniform(*AREA_MIN_MAX) for _ in range(NUM_DIMS)])
                      for _ in range(NUM_AREAS)]

    area_centroids = np.array(area_centroids)

    # create whole population
    population = []
    for c in area_centroids:
        # create random points around the centroid c
        area_points = [(tuple([random.gauss(xc, 1.0) for xc in c])) for _ in range(NUM_POINTS_PER_AREA)]
        population.extend(area_points)

    population = np.array(population)
    print(f"Population shape: {population.shape}")

    plt.title("Generated data")
    # display the population as blue crosses
    plt.scatter(population[:, 0], population[:, 1], marker='x', color='blue')
    # display random centroids as orange circles
    plt.scatter(area_centroids[:, 0], area_centroids[:, 1], marker='o', color='orange', label='Generated Centroids')
    plt.legend()
    plt.show()

    # generate codebook
    start = time.time()
    cb, labels, cb_abs_w, cb_rel_w = generate_codebook(population, SIZE_CODEBOOK)
    print(f'LGB took {time.time() - start:.3f}s')

    label_color = [LABEL_COLOR_MAP[l] for l in labels]

    plt.title("Labelled data")
    plt.scatter(population[:, 0], population[:, 1], marker='x', color=label_color)

    plt.scatter(cb[:, 0], cb[:, 1], s=[((w + 1) ** 5) * 40 for w in cb_rel_w], marker='o', color='red', label='Found Centroids')
    plt.legend()
    plt.show()


def lgb_complex_example():
    import time
    import random
    import matplotlib.pyplot as plt
    from sklearn.decomposition import PCA

    NUM_AREAS = SIZE_CODEBOOK = 8
    NUM_DIMS = 5
    NUM_POINTS_PER_AREA = 1000
    AREA_MIN_MAX = (-20, 20)

    random.seed(0)

    # create random centroids for NUM_AREAS areas
    area_centroids = [tuple([random.uniform(*AREA_MIN_MAX) for _ in range(NUM_DIMS)])
                      for _ in range(NUM_AREAS)]

    area_centroids = np.array(area_centroids)

    # create whole population
    population = []
    for c in area_centroids:
        # create random points around the centroid c
        area_points = [(tuple([random.gauss(xc, 1.0) for xc in c])) for _ in range(NUM_POINTS_PER_AREA)]
        population.extend(area_points)

    population = np.array(population)
    print(f"Population shape: {population.shape}")

    pca_ = PCA(n_components=2)
    population_pca = pca_.fit_transform(population)
    area_centroids_pca = pca_.transform(area_centroids)

    plt.title("Generated data")
    # display the population as blue crosses
    plt.scatter(population_pca[:, 0], population_pca[:, 1], marker='x', color='blue')
    # display random centroids as orange circles
    plt.scatter(area_centroids_pca[:, 0], area_centroids_pca[:, 1], marker='o', color='orange', label='Generated Centroids')
    plt.legend()
    plt.show()

    # generate codebook
    start = time.time()
    cb, labels, cb_abs_w, cb_rel_w = generate_codebook(population, SIZE_CODEBOOK)
    print(f'LGB took {time.time() - start:.3f}s')

    label_color = [LABEL_COLOR_MAP[l] for l in labels]
    cb_pca = pca_.transform(cb)

    plt.title("Labelled data")
    plt.scatter(population_pca[:, 0], population_pca[:, 1], marker='x', color=label_color)

    plt.scatter(cb_pca[:, 0], cb_pca[:, 1], s=[((w + 1) ** 5) * 40 for w in cb_rel_w], marker='o', color='red', label='Found Centroids')
    plt.legend()
    plt.show()


if __name__ == '__main__':
    lgb_simple_example()
    # lgb_complex_example()