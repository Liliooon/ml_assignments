#!/usr/bin/env python
# coding: utf-8

# ### K-means

# 1. 
# Write a clustering problem generator. For k=3, generate easy and hard problems and plot them; the easy problem might look like figure 3.13 from Daumé.

# In[38]:


import numpy as np
from random import choice
import matplotlib.pyplot as plt
from math import dist
from statistics import mean


# In[57]:


def scatter_clusters(
    centers: list,
    spread: list,
    n_points: int
) -> dict:
    
    points = []
    clusters = {center: [] for center in centers}
    
    for point in range(n_points):
        point_center = choice(centers)
        point_dist_x = choice(spread)
        point_dist_y = choice(spread)
        x = point_center[0] + point_dist_x
        y = point_center[1] + point_dist_y
        point_coord = (x, y)
        points.append(point_coord)
        clusters[point_center].append(point_coord)
    
    points_clusters = {'points': points, 'clusters': clusters}
        
    return points_clusters


# In[58]:


def plot_scattering(scattering):
    for x, y in scattering:
        plt.scatter(x, y, c='black')
    plt.show()


# In[64]:


centers = [(1, 2), (20, 80), (50, 60)]
n_points = 60


# In[65]:


easy_problem = scatter_clusters(centers, list(range(15)), n_points)


# In[66]:


hard_problem = scatter_clusters(centers, list(range(50)), n_points)


# In[69]:


plot_scattering(easy_problem['points'])
plt.show()


# In[71]:


plot_scattering(hard_problem['points'])
plt.show()


# 2. Implement K-means clustering as shown in Daumé. Replot your problems at 5 stages (random initialisation, 25%, 50%, 75%, 100% of iterations), using colours to assign points to clusters. The easy problem plots might look like the coloured plots in figure 3.14 from Daumé.

# In[73]:


def kmeans_cluster_assignment(
    k: int,
    points: list,
    centers_guess: list = None,
    max_iterations: int = None,
    tolerance: float = None
) -> dict:

#     randomly initialize center for kth cluster

    max_x = max([point[0] for point in points])
    max_y = max([point[1] for point in points])

    if centers_guess:
        assignment_guess = {center: [] for center in centers_guess}
    else:
        assignment_guess = {}
        for cluster in range(k):
            random_center = (choice(range(max_x)), choice(range(max_y)))
            assignment_guess[random_center] = []

    n_iteration = 0

    while True:

        n_iteration += 1

#     assign points to cluster k

        for point in points:
            point_to_center_dists = [(center, dist(center, point))
                                     for center in assignment_guess]
            point_center = min(point_to_center_dists, key=lambda x: x[1])[0]
            assignment_guess[point_center].append(point)

#     re-estimate center of cluster k

        centers_guess = []

        for center, center_points in assignment_guess.items():
            if center_points:
                x = mean([point[0] for point in center_points])
                y = mean([point[1] for point in center_points])
                centers_guess.append((x, y))
            else:
                centers_guess.append(
                    (choice(range(max_x)), choice(range(max_y))))

        if max_iterations:
            if n_iteration == max_iterations:
                break
        if tolerance:
            center_dists = [dist(cur_center, ex_center) for cur_center in centers_guess
                            for ex_center in assignment_guess]
            if all(dist <= tolerance for dist in sorted(center_dists)[:k]):
                break
            else:
                assignment_guess = {center: [] for center in centers_guess}
        elif set(centers_guess) == set(assignment_guess):
            break
        else:
            assignment_guess = {center: [] for center in centers_guess}

#     return cluster assignments

    assignment = {center: [] for center in centers_guess}
    for point in points:
        point_to_center_dists = [(center, dist(center, point))
                                 for center in centers_guess]
        point_center = min(point_to_center_dists, key=lambda x: x[1])[0]
        assignment[point_center].append(point)
    
    return assignment


# In[74]:


def plot_scattering_with_color(assignment):
    colors = plt.cm.rainbow(np.linspace(0, 1, len(assignment)))
    for cluster, color in zip(assignment, colors):
        for x, y in assignment[cluster]:
            plt.scatter(x, y, c=color.reshape(1,-1))
    plt.show()


# In[90]:


k = 3


# In[91]:


# random initialisation

easy_problem_solution = kmeans_cluster_assignment(k, easy_problem['points']) 

hard_problem_solution = kmeans_cluster_assignment(k, hard_problem['points'])

plot_scattering_with_color(easy_problem_solution)
plt.show()

plot_scattering_with_color(hard_problem_solution)
plt.show()


# In[92]:


# 25%

max_iter = 25

easy_problem_solution = kmeans_cluster_assignment(k, easy_problem['points'], max_iterations=max_iter) 

hard_problem_solution = kmeans_cluster_assignment(k, hard_problem['points'], max_iterations=max_iter)

plot_scattering_with_color(easy_problem_solution)
plt.show()

plot_scattering_with_color(hard_problem_solution)
plt.show()


# In[93]:


# 50%

max_iter = 50

easy_problem_solution = kmeans_cluster_assignment(k, easy_problem['points'], max_iterations=max_iter) 

hard_problem_solution = kmeans_cluster_assignment(k, hard_problem['points'], max_iterations=max_iter)

plot_scattering_with_color(easy_problem_solution)
plt.show()

plot_scattering_with_color(hard_problem_solution)
plt.show()


# In[94]:


# 75%

max_iter = 75

easy_problem_solution = kmeans_cluster_assignment(k, easy_problem['points'], max_iterations=max_iter) 

hard_problem_solution = kmeans_cluster_assignment(k, hard_problem['points'], max_iterations=max_iter)

plot_scattering_with_color(easy_problem_solution)
plt.show()

plot_scattering_with_color(hard_problem_solution)
plt.show()


# In[95]:


# 100%

max_iter = 100

easy_problem_solution = kmeans_cluster_assignment(k, easy_problem['points'], max_iterations=max_iter) 

hard_problem_solution = kmeans_cluster_assignment(k, hard_problem['points'], max_iterations=max_iter)

plot_scattering_with_color(easy_problem_solution)
plt.show()

plot_scattering_with_color(hard_problem_solution)
plt.show()


# In[96]:


# to be continued