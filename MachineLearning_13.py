import numpy as np
import matplotlib.pyplot as plt
import time

x = [
    (126, 63), (101, 100), (80, 160), (88, 208), (89, 282),
    (88, 362), (94, 406), (149, 377), (147, 304), (147, 235),
    (146, 152), (160, 103), (174, 142), (169, 184), (170, 241),
    (169, 293), (185, 376), (178, 422), (116, 353), (124, 194),
    (273, 69), (277, 112), (260, 150), (265, 185), (270, 235),
    (265, 295), (281, 351), (285, 416), (321, 404), (316, 366),
    (306, 304), (309, 254), (309, 207), (327, 161), (318, 108),
    (306, 66), (425, 66), (418, 135), (411, 183), (413, 243),
    (414, 285), (407, 333), (411, 385), (443, 387), (455, 330),
    (441, 252), (457, 207), (453, 149), (455, 90), (455, 56),
    (439, 102), (431, 162), (431, 193), (426, 236), (427, 281),
    (438, 323), (419, 379), (425, 389), (422, 349), (451, 275),
    (441, 222), (297, 145), (284, 195), (288, 237), (292, 282),
    (288, 313), (303, 356), (293, 395), (274, 268), (280, 344),
    (303, 187), (114, 247), (131, 270), (144, 215), (124, 219),
    (98, 240), (96, 281), (146, 267), (136, 221), (123, 166),
    (101, 185), (152, 184), (104, 283), (74, 239), (107, 287),
    (118, 335), (89, 336), (91, 315), (151, 340), (131, 373),
    (108, 133), (134, 130), (94, 260), (113, 193)
]

M = np.mean(x, axis=0)
D = np.var(x, axis=0)
K = 3
ma = [np.random.normal(M, np.sqrt(D / 10), 2) for n in range(K)]


def ro(x_vect, m_vect):
    return np.mean((x_vect - m_vect) ** 2)


COLORS = ('green', 'blue', 'brown', 'black')

plt.ion()
n = 0
while n < 10:
    X = [[] for i in range(K)]

    for x_vect in x:
        r = [ro(x_vect, m) for m in ma]
        X[np.argmin(r)].append(x_vect)

    ma = [np.mean(xx, axis=0) for xx in X]

    plt.clf()

    for i in range(K):
        xx = np.array(X[i]).T
        plt.scatter(xx[0], xx[1], s=10, color=COLORS[i])

    mx = [m[0] for m in ma]
    my = [m[1] for m in ma]
    plt.scatter(mx, my, s=50, color='red')

    plt.draw()
    plt.gcf().canvas.flush_events()
    time.sleep(0.2)

    n += 1

plt.ioff()

for i in range(K):
    xx = np.array(X[i]).T
    plt.scatter(xx[0], xx[1], s=10, color=COLORS[i])

mx = [m[0] for m in ma]
my = [m[1] for m in ma]
plt.scatter(mx, my, s=50, color='red')

plt.show()
