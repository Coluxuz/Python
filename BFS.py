#list
from queue import Queue
adj_list = {
    "A":["B","C"],
    "B":["A","D","E"],
    "C":["A","F","G"],
    "D":["B","H","I"],
    "E":["B","J","K"],
    "F":["C","L","M"],
    "G":["C","N","O"],
    "H":["D"],
    "I":["D"],
    "J":["E"],
    "K":["E","P","Q"],
    "L":["F"],
    "M":["F"],
    "N":["G"],
    "O":["G"],
    "P":["K"],
    "Q":["K"],
}

#bfs code
visited = {}
level = {}
parent = {}
bfs_traversal_output = []
queue = Queue()

for node in adj_list.keys():
    visited[node] = False
    parent[node] = None
    level[node] = -1
   
   
s = "A"
visited[s] = True
level[s] = 0    
queue.put(s)

while not queue.empty():
    u = queue.get()
    bfs_traversal_output.append(u)
    
    for v in adj_list[u]:
        if not visited[v]:
            visited[v] = True
            parent[v] = u
            level[v] = level[u]+1
            queue.put(v)

print("\nBFS output is: ",bfs_traversal_output)

#shortest path of from any node from source node 
x=input("enter final node to find shortest path:")
v = x
path = []
while v is not None:
    path.append(v)
    v = parent[v]
path.reverse()
print("shortest distance is: ",path)
z=input("enter node that you want to find level:")
print("level is :",z,level[z])
