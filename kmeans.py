##########################################################################
# DESCRIPTION:
# This program shows the results of using the kmeans++ algoritm # 
# (http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) in comparison to the 
# standard k-means algorithm. Results are shown graphically when run 
# with the dimension parameter set to "2". In all cases the program prints
# the number of iterations to convergance and the squared error of the point
# set.
# 
# ARGUMENTS:
# 1) total number of points
# 2) number of dimensions / degrees of freedom for points 
# 3) k (number of groupings)
# 4) init method (1=normal kmeans, 2=kmeans++)
# 5) random seed value for selecting points
# 6) random seed value for selecting start centers
# 7) name for exported (png) image file
#
# EXAMPLES:
# The points are the same for these (since the random seed is "7"). 
# Changing the random seed for the center start points, we see that kmeans++ 
# consistently has better groupings.
#
# The first of these pairings is standard; the second is k-means++ (see arguments):
#
# python3 kmeans.py 300 2 5 1 7 2 result_standard.png
# python3 kmeans.py 300 2 5 2 7 2 result_plus_plus.png
#
# python3 kmeans.py 300 2 5 1 7 3 result_standard.png
# python3 kmeans.py 300 2 5 2 7 3 result_plus_plus.png
#
# python3 kmeans.py 300 2 5 1 7 5 result_standard.png
# python3 kmeans.py 300 2 5 2 7 5 result_plus_plus.png
##########################################################################

import sys
import numpy as np
import matplotlib.pyplot as plt

MAX_VALUE = 100.0
SPREAD_VALUE = 30.0

adds = 0   # this has global scope
iters = 0   # this has global scope

# This method takes an np array and returns an np array
# of k randomly chosen start points
def randomInit(points,k):
    chosen = []
    centers = np.zeros((k,points.shape[1]),dtype=float)
    for i in range(0,k):
        while True:
            x = np.random.randint(points.shape[0])
            if x in chosen:
                continue
            centers[i] = points[x]
            chosen.append(x)
            break
    return centers

# This method takes an np array and returns an np array
# of k++ chosen centers
def kPlusPlusInit(points,k):
    global adds
    chosen = []
    centers = np.zeros((k,points.shape[1]),dtype=float)
    chosen.append(np.random.randint(points.shape[0]))
    centers[0] = points[chosen[0]]
    min_total_sq_dist = 99999.9*np.ones(points.shape[0])

    for i in range(1,k):
        min_total_sq_dist = 99999.9*np.ones(points.shape[0])
        for j in range(points.shape[0]):
            if j in chosen:
                continue
            for k in range(len(chosen)):
                cur_dist = 0.0
                for d in range(points.shape[1]):
                    cur_dist += np.square(centers[k,d]-points[j,d])
                    adds += 1
                if cur_dist < min_total_sq_dist[j]:
                    min_total_sq_dist[j] = cur_dist
        sum_min_total_sq_dist = sum(min_total_sq_dist)
        adds += len(min_total_sq_dist)
        while True:
            x = np.random.randint(points.shape[0])
            if x in chosen or np.random.rand() > min_total_sq_dist[x]/sum_min_total_sq_dist:
                continue
            centers[i] = points[x]
            chosen.append(x)
            break
    
    return centers

# This is the clustering algorithm that assigns all points to a center
def kMeans(points,centers):
    global adds 
    global iters
    assignments = [-1]*points.shape[0]                      #cluster assignments
    prev_centers = np.zeros(centers.shape,dtype=float)
    sq_dist = np.zeros(points.shape[0],dtype=float)         #squared distance from point to each center

    # Assign points to a cluster
    while( np.array_equal( centers, prev_centers ) == False ):
        for i in range(points.shape[0]):
            for c in range(centers.shape[0]):
                cur_dist = 0.0
                for d in range(points.shape[1]):
                    cur_dist += np.square(centers[c,d]-points[i,d])
                    adds += 1
                if assignments[i] == -1 or cur_dist < sq_dist[i]:
                    assignments[i] = c
                    sq_dist[i] = cur_dist
        
        # Update cluster centers
        prev_centers = np.copy( centers )
        centers = np.zeros(centers.shape,dtype=float)
        for d in range(points.shape[1]):
            cluster_counts = np.zeros(centers.shape[0])
            for i in range(points.shape[0]):
                cluster_counts[assignments[i]] += 1
                centers[assignments[i],d] += points[i,d]
                adds += 1
            for c in range(centers.shape[0]):
                centers[c,d] /= cluster_counts[c]
        iters += 1

    return assignments, centers

# The main method is the entry point for the program
def main():
    global iters

    # Get command line args
    if len(sys.argv) != 8:
        print('Must provide arguments for n, d, k, initialization method, points seed, and centers seed')
        return

    n = int(sys.argv[1])
    d = int(sys.argv[2])
    k = int(sys.argv[3])
    init_method = int(sys.argv[4])
    points_seed = int(sys.argv[5])
    centers_seed = int(sys.argv[6])
    image_path = sys.argv[7]
    points_per_cluster = int(max(1,(n-k)/k))

    np.random.seed(points_seed)
    points = np.random.uniform(0,MAX_VALUE,(k+k*points_per_cluster,d))

    # Use first k points as artificial centers around which the algorithm should create ideal clusters
    cur_point = k
    group_start_point = k
    for i in range(k):
        for dim in range(d):
            cur_point = group_start_point
            min_value = max(0,points[i][dim]-(SPREAD_VALUE/2))
            max_value = min(100,points[i][dim]+(SPREAD_VALUE/2))
            for j in range(points_per_cluster):
                points[cur_point][dim] = (max_value - min_value) * np.random.random_sample() + min_value
                cur_point += 1
        group_start_point = cur_point

    # Initialize either to random points or points chosen by the k-means++ algorithm
    np.random.seed(centers_seed)
    if init_method == 1:
        centers = randomInit(points,k)
    else:
        centers = kPlusPlusInit(points,k)

    # Assign points to clusters with k-means algoritm
    assignments, centers = kMeans(points,centers)
    # Calculate scaled squared error
    sq_err = 0.0
    for i in range(points.shape[0]):
        sq_err += pow((points[i]-centers[assignments[i]]),2)/points.shape[0]

    # Use this print for analysis
    print(iters,sq_err[0])

    '''
    n = points.shape[0];
    print('Iterations: ',iters)
    print('n: ',n)
    print('d: ',d)
    print('k: ',k)
    print('Expected kmeans adds: ',iters*n*d*k + iters*n*d)
    print('Expected kmeans++ adds: ',d*(n*k*(k-1)/2 - k*(k-1)*(2*k-1)/6) + n*(k-1))
    print('Additions: ',adds)
    print('Error: ',sq_err[0])
    '''
    
    # Plot results if this is 2d
    if d == 2 and image_path != "":
        plt.scatter(points[:,0],points[:,1],c=assignments)
        plt.savefig( image_path )

main()
