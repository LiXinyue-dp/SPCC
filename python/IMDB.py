#!/usr/bin/env python3
import csv
import sys
import numpy as np
from scipy.sparse import lil_matrix
import random


IMDBFile = "../data//IMDB.mtx"
EdgeFile = "../data//IMDB/edges.csv" 
DegFile = "../data/IMDB/deg.csv"


def ReadIMDB():
    movie_actor_lst = []
    
    with open(IMDBFile, "r") as f:
        for i, line in enumerate(f):
            if i < 55:
                continue
            elif i == 55:
                lst = line.rstrip("\n").split(" ")
                movie_num = int(lst[0])
                actor_num = int(lst[1])
            else:
                lst = line.rstrip("\n").split(" ")
                movie_id = int(lst[0]) - 1  # Convert to 0-based index
                actor_id = int(lst[1]) - 1
                movie_actor_lst.append([movie_id, actor_id])
    
    return movie_actor_lst, movie_num, actor_num


# Read the IMDB file
movie_actor_lst, movie_num, actor_num = ReadIMDB()

# Make a movie dictionary {movie_id: [actor_ids]}
movie_dic = {i: [] for i in range(movie_num)}
for movie_id, actor_id in movie_actor_lst:
    movie_dic[movie_id].append(actor_id)

# Step 1: Use all actors (no sampling)
all_actors = set()
for actors in movie_dic.values():
    all_actors.update(actors)
total_actors = len(all_actors)
print(f"Total actors: {total_actors}")

# Create mapping from original actor IDs to new contiguous IDs (0 to total_actors-1)
actor_id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(all_actors))}

# Step 2: Build the full graph edges and degrees
print("Building full graph edges...")
edges_lil = lil_matrix((total_actors, total_actors))
degrees = np.zeros(total_actors)

for movie_id, actors in movie_dic.items():
    # Create all possible edges between actors in this movie
    for i in range(len(actors)):
        for j in range(i + 1, len(actors)):
            actor1 = actor_id_map[actors[i]]
            actor2 = actor_id_map[actors[j]]
            
            if edges_lil[actor1, actor2] == 0:
                edges_lil[actor1, actor2] = 1
                degrees[actor1] += 1
                degrees[actor2] += 1

# Get the edges in COO format
rows, cols = edges_lil.nonzero()
print(f"Full graph has {len(rows)} edges and {total_actors} nodes")

# Output edge information
print("Writing edge file...")
with open(EdgeFile, "w") as f:
    print("#nodes", file=f)
    print(total_actors, file=f)
    print("node,node", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for i in range(len(rows)):
        writer.writerow([rows[i], cols[i]])

# Output degree information
print("Writing degree file...")
with open(DegFile, "w") as f:
    print("node,deg", file=f)
    writer = csv.writer(f, lineterminator="\n")
    for node in range(total_actors):
        writer.writerow([node, int(degrees[node])])

print("Done! Full graph with", total_actors, "nodes created.")