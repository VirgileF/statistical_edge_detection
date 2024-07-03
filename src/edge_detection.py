import numpy as np
import scipy
import scipy.stats
from scipy.stats import norm

from sklearn.neighbors import KDTree

import time as T

def compute_frechet(X, bounds=(0,1), precision=1e-3):
    
    # reshape to column vector
    X = X.reshape(-1,1)
    
    # grid search
    candidates = np.arange(bounds[0], bounds[1]+precision, precision).reshape(1,-1)
    
    # compute sum of squared distances (ssd) to each of the candidates    
    ssd_X_to_candidates = np.sum(np.minimum(np.abs(X-candidates), 1-np.abs(X-candidates))**2, axis=0)

    # find the index of Fréchet
    mF_index = np.argmin(ssd_X_to_candidates)

    # deduce best approximate of Fréchet mean among candidates
    mF = candidates[0,mF_index]
    
    return mF

def compute_neighborhoods(
    point_cloud=None, 
    adjacency_graph=None, 
    method='nn', 
    k_neighbors=None, 
    radius=None
    ):

    """
    Given a point cloud or its adjacency graph, this function computes the neighborhood of each point
    w.r.t. a method.

    Parameters:
    -----------
    point_cloud: arr (n_points, dim) (default: None)
        The set of points representing a surface.
    adjacency_graph: networkx.Graph object or None (default: None)
        The adjacency graph of the points. Can be extracted from a 3D-mesh. Necessary input 
        for 'mesh-connectivity' method.
    method: str (default: 'nn')
        - 'nn': select k nearest-neighbors in the point cloud 
        - 'radius': select all neighbors that are inside a sphere of a given radius
        - 'mesh-connectivity': select all neighbors and neighbors' neighbors w.r.t. the graph connectivity
        - 'parametric-neighboring': works only for 2D oriented contours (points are well ordered)
    k_neighbors: int or None (default: None)
        For 'nn' method.
    radius: float or None (default: None)
        For 'radius' method.

    Returns:
    --------
    neighborhoods_list: list of lists of integers (neighbors in each neighborhood)

    """

    neighborhoods_list = []
    
    if method == 'nn':
        
        assert point_cloud is not None
        assert k_neighbors is not None

        # index point cloud in KdTree
        kdtree = KDTree(point_cloud)
        neighborhoods_list = kdtree.query(point_cloud, k=k_neighbors, return_distance=False).tolist()
        neighborhoods_list = [np.array(neighborhood) for neighborhood in neighborhoods_list]
    
    elif method == 'radius':
        
        assert point_cloud is not None
        assert radius is not None
        kdtree = KDTree(point_cloud)
        neighborhoods_list = kdtree.query_radius(point_cloud, r=radius, return_distance=False).tolist()

    elif method == 'mesh-connectivity':
        
        assert adjacency_graph is not None
        for i in range(len(adjacency_graph)):
            neighbors = list(adjacency_graph[i])
            for n in list(adjacency_graph[i]):
                neighbors += list(adjacency_graph[n])
            neighborhoods_list.append(list(set(neighbors)))

    elif method == 'parametric-neighboring':
        
        assert point_cloud.shape[1] == 2
        assert k_neighbors is not None
        
        n_points = point_cloud.shape[0] 
        
        for i in range(point_cloud.shape[0]):
            if i-k_neighbors//2<0:
                neighborhoods_list.append(np.concatenate(
                    (
                        np.arange((i-k_neighbors//2)%n_points, n_points), 
                        np.arange(0,i+k_neighbors//2+1)
                    ), 
                    axis=0))
            elif i+k_neighbors//2>=n_points:
                neighborhoods_list.append(np.concatenate(
                    (
                        np.arange(i-k_neighbors//2, n_points), 
                        np.arange(0,(i+k_neighbors//2+1)%n_points)
                    ),
                    axis=0))
            else:
                neighborhoods_list.append(np.arange(i-k_neighbors//2,i+k_neighbors//2+1))
    
    else:
        raise TypeError('Invalid neighboring method.')
        
    return neighborhoods_list

def compute_odds_pvalue(X, centroid):
    
    # covariance matrix
    V = (X-X.mean(0)).T @ (X-X.mean(0))
    
    # eigen analysis
    lambdas, es = np.linalg.eig(V)
    e_1 = es[:,lambdas.argmax()].reshape(-1,1)
    
    # projection
    X_proj = (X-centroid) @ e_1
    
    # test statistic
    k = X.shape[0]-1
    k_plus = np.sum(X_proj>0)
    v = (k_plus - (k-k_plus))/np.sqrt(k)
    
    # p-value
    p_value = 2*(1-norm.cdf(np.abs(v)))
    
    return p_value

def compute_pauly_descriptor(X):
    
    # covariance matrix
    V = (X-X.mean(0)).T @ (X-X.mean(0))
    
    # eigen analysis
    lambdas, _ = np.linalg.eig(V)
    
    # Variation ratio
    ratio = lambdas.min()/lambdas.sum()

    # Rescaling [0,1/D] -> [0,1] where D is the dimension of the ambient space
    dim = X.shape[1]
    descriptor = ratio * dim
    
    return descriptor

def compute_pauly_descriptor_on_surface(point_cloud, neighborhoods_list):

    t0 = T.time()

    descriptors = np.zeros(len(neighborhoods_list))
    for i, indices in enumerate(neighborhoods_list):

        X = point_cloud[indices]
        descriptors[i] = compute_pauly_descriptor(X)
    
    return descriptors

def compute_ks_p_value(X, centroid):
    
    # compute covariance matrix
    V = (X-X.mean(0)).T @ (X-X.mean(0))
    
    # eigenanalysis
    lambdas, es = np.linalg.eig(V)
    
    # projection onto the average plane
    es_1_2 = es[:,lambdas.argsort()[::-1][:2]]
    X_proj = (X-centroid) @ es_1_2

    # polar coordinates
    angles = np.angle(X_proj[:,0] + 1j*X_proj[:,1])
    angles = angles/(2*np.pi) + .5
    
    # frechet centering
    mF = compute_frechet(angles)
    angles = np.mod(angles - (mF - .5), 1)
    
    # KS test
    _, p_value = scipy.stats.kstest(angles, cdf='uniform')
    
    return p_value

def compute_ks_pvalues_on_surface(point_cloud, neighborhoods_list):

    t0 = T.time()

    descriptors = np.zeros(len(neighborhoods_list))
    for i, indices in enumerate(neighborhoods_list):

        centroid = point_cloud[i]
        X = point_cloud[indices]
        descriptors[i] = compute_ks_p_value(X, centroid)
    
    return descriptors
