from math import inf
import heapq
n,m,y = map(int,input().split())
adjlist = [[] for _ in range(n)]
for _ in range(m):
    u, v, t = map(int, input().split())
    adjlist[u - 1].append((v - 1, t))
    adjlist[v - 1].append((u - 1, t))

warpgates = list(map(int,input().split()))

def djikstra(adjlist, s):
    dist = [inf] * n
    vheap = []
    dist[s] = 0
    heapq.heappush(vheap, (0, s))
    while vheap:
        currentdist, v = heapq.heappop(vheap)
        if dist[v] < currentdist:
            continue
        for i in range(len(adjlist[v])):
            neighbour = adjlist[v][i][0]
            weight = adjlist[v][i][1]
            if dist[neighbour] > dist[v] + weight:
                newdist = dist[v] + weight
                dist[neighbour] = newdist
                heapq.heappush(vheap, (newdist, neighbour))
    return dist


distances = djikstra(adjlist, 0)
res = []
for i in range(1, n):
    warpgatedist = warpgates[0]+warpgates[i]+y
    res.append(min(distances[i],warpgatedist))

print(" ".join(map(str,res)))

# distances = djikstra(adjlist,0)
