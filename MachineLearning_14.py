from itertools import cycle
from math import hypot
from numpy import random
import matplotlib.pyplot as plt


def dbscan_naive(P, eps, m, distance):
    NOISE = 0
    C = 0

    visited_points = set()
    clustered_points = set()
    clusters = {NOISE: []}

    def region_query(p):
        return [q for q in P if distance(p, q) < eps]

    def expand_cluster(p, neighbours):
        if C not in clusters:
            clusters[C] = []

        clusters[C].append(p)
        clustered_points.add(p)

        while neighbours:
            q = neighbours.pop()

            if q not in visited_points:
                visited_points.add(q)
                neighbourz = region_query(q)

                if len(neighbourz) > m:
                    neighbours.extend(neighbourz)

            if q not in clustered_points:
                clustered_points.add(q)
                clusters[C].append(q)

                if q in clusters[NOISE]:
                    clusters[NOISE].remove(q)

    for p in P:
        if p in visited_points:
            continue

        visited_points.add(p)
        neighbours = region_query(p)

        if len(neighbours) < m:
            clusters[NOISE].append(p)
        else:
            C += 1
            expand_cluster(p, neighbours)

    return clusters


P = [
    (64, 150), (84, 112), (106, 90), (154, 64), (192, 62),
    (220, 82), (244, 92), (271, 111), (275, 137), (286, 161),
    (56, 178), (80, 156), (101, 131), (123, 104), (155, 94),
    (191, 100), (242, 70), (231, 114), (272, 95), (261, 131),
    (299, 136), (308, 124), (128, 78), (47, 128), (47, 159),
    (137, 186), (166, 228), (171, 250), (194, 272), (221, 287),
    (253, 292), (308, 293), (332, 280), (385, 256), (398, 237),
    (413, 205), (435, 166), (447, 137), (422, 126), (400, 154),
    (389, 183), (374, 214), (358, 235), (321, 250), (274, 263),
    (249, 263), (208, 230), (192, 204), (182, 174), (147, 205),
    (136, 246), (147, 255), (182, 282), (204, 298), (252, 316),
    (312, 321), (349, 313), (393, 288), (417, 259), (434, 222),
    (443, 187), (463, 174)
]

eps = 60
m = 5

clusters = dbscan_naive(
    P,
    eps,
    m,
    lambda x, y: hypot(x[0] - y[0], x[1] - y[1])
)

for c, points in zip(
    cycle('bgrcmykgrcmykgrcmykgrcmykgrcmykgrcmyk'), clusters.values()
):
    X = [p[0] for p in points]
    Y = [p[1] for p in points]
    plt.scatter(X, Y, c=c)

plt.show()
