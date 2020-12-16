# k-means_plus_plus
Comparison of standard k-means algorithm vs. k-means++

# Summary
This program shows the results of using the kmeans++ algoritm
(http://ilpubs.stanford.edu:8090/778/1/2006-13.pdf) in comparison
to the standard kmeans algorithm. Results are shown graphically,as 
when run with the arguments below, and are also compiled for a
quantitative comparison of result sets in the Results directory.

# Arguments
Args: 
    total number of points, 
    number of dimensions / degrees of freedom for points, 
    k (number of groupings), 
    init method (1=normal kmeans, 2=kmeans++), 
    random seed value for selecting points, 
    random seed value for selecting start centers

# Examples
The points are the same for these (since the random seed is "7").
Changing the random seed for the center start points, we see that 
kmeans++ consistently has better groupings.

The first of these pairings is standard; the second is k-means++
(see arguments)

python3 kmeans.py 300 2 5 1 7 2
python3 kmeans.py 300 2 5 2 7 2

python3 kmeans.py 300 2 5 1 7 3
python3 kmeans.py 300 2 5 2 7 3

python3 kmeans.py 300 2 5 1 7 5
python3 kmeans.py 300 2 5 2 7 5

