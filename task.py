import sys
import math
import numpy as np
from sklearn.cluster import KMeans


class ClusterStat(object):
    def __init__(self, points=None, point_ids=None):
        if points is None:
            self.N = 0
            self.SUM = None
            self.SUMSQ = None
            self.point_ids = []
        else:
            pts = np.asarray(points, dtype=np.float64)
            self.N = int(pts.shape[0])
            self.SUM = np.sum(pts, axis=0)
            self.SUMSQ = np.sum(pts * pts, axis=0)
            self.point_ids = [int(x) for x in point_ids]

    def add_point(self, point, point_id):
        p = np.asarray(point, dtype=np.float64)
        if self.N == 0:
            self.SUM = np.zeros_like(p)
            self.SUMSQ = np.zeros_like(p)
        self.N += 1
        self.SUM += p
        self.SUMSQ += p * p
        self.point_ids.append(int(point_id))

    def merge(self, other):
        if self.N == 0:
            self.N = other.N
            self.SUM = other.SUM.copy()
            self.SUMSQ = other.SUMSQ.copy()
            self.point_ids = list(other.point_ids)
        else:
            self.N += other.N
            self.SUM += other.SUM
            self.SUMSQ += other.SUMSQ
            self.point_ids.extend(other.point_ids)

    def centroid(self):
        return self.SUM / float(self.N)

    def std(self):
        mu = self.centroid()
        var = self.SUMSQ / float(self.N) - mu * mu
        return np.sqrt(np.maximum(var, 1e-6))


def mahalanobis_point(point, cluster):
    mu = cluster.centroid()
    s = cluster.std()
    diff = (np.asarray(point, dtype=np.float64) - mu) / s
    return float(np.sqrt(np.sum(diff * diff)))


def mahalanobis_clusters(c1, c2):
    d1 = mahalanobis_point(c1.centroid(), c2)
    d2 = mahalanobis_point(c2.centroid(), c1)
    return min(d1, d2)


def load_data(path):
    indices = []
    features = []
    with open(path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(',')
            indices.append(int(float(parts[0])))
            features.append([float(x) for x in parts[2:]])
    return np.asarray(indices, dtype=np.int64), np.asarray(features, dtype=np.float64)


def run_kmeans(points, k, random_state=553):
    n = len(points)
    if n == 0:
        return np.array([], dtype=np.int64)
    k = max(1, min(int(k), n))
    try:
        km = KMeans(n_clusters=k, random_state=random_state, n_init=10)
    except TypeError:
        km = KMeans(n_clusters=k, random_state=random_state)
    return km.fit_predict(np.asarray(points, dtype=np.float64))


def split_by_size(labels, point_ids, points):
    groups = {}
    for i, lab in enumerate(labels):
        groups.setdefault(int(lab), []).append(i)
    multi = []
    singles = []
    for idxs in groups.values():
        if len(idxs) > 1:
            multi.append(([int(point_ids[j]) for j in idxs], points[idxs]))
        else:
            j = idxs[0]
            singles.append((int(point_ids[j]), points[j]))
    return multi, singles


def rs_to_cs_and_rs(rs_ids, rs_points, n_cluster, random_state=553):
    cs_list = []
    new_rs_ids = []
    new_rs_points = []
    if len(rs_ids) == 0:
        return cs_list, new_rs_ids, new_rs_points
    rs_points = np.asarray(rs_points, dtype=np.float64)
    if len(rs_ids) == 1:
        return cs_list, [int(rs_ids[0])], [rs_points[0]]

    large_k = min(5 * n_cluster, max(1, int(math.ceil(len(rs_ids) / 2.0))))
    labels = run_kmeans(rs_points, large_k, random_state=random_state)
    multi, singles = split_by_size(labels, rs_ids, rs_points)
    for ids, pts in multi:
        cs_list.append(ClusterStat(pts, ids))
    for pid, pt in singles:
        new_rs_ids.append(int(pid))
        new_rs_points.append(pt)
    return cs_list, new_rs_ids, new_rs_points


def assign_to_nearest(points, point_ids, clusters, threshold):
    if len(clusters) == 0 or len(point_ids) == 0:
        return [int(x) for x in point_ids], list(points)
    remain_ids = []
    remain_pts = []
    for pid, pt in zip(point_ids, points):
        best_i = -1
        best_d = float('inf')
        for i, c in enumerate(clusters):
            d = mahalanobis_point(pt, c)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d < threshold:
            clusters[best_i].add_point(pt, pid)
        else:
            remain_ids.append(int(pid))
            remain_pts.append(pt)
    return remain_ids, remain_pts


def merge_cs_clusters(cs_list, threshold):
    if len(cs_list) <= 1:
        return cs_list
    merged = True
    while merged:
        merged = False
        n = len(cs_list)
        used = [False] * n
        new_list = []
        for i in range(n):
            if used[i]:
                continue
            best_j = -1
            best_d = float('inf')
            for j in range(i + 1, n):
                if used[j]:
                    continue
                d = mahalanobis_clusters(cs_list[i], cs_list[j])
                if d < threshold and d < best_d:
                    best_d = d
                    best_j = j
            if best_j >= 0:
                cs_list[i].merge(cs_list[best_j])
                used[best_j] = True
                merged = True
            new_list.append(cs_list[i])
            used[i] = True
        cs_list = new_list
    return cs_list


def merge_cs_into_ds(ds_list, cs_list, threshold):
    if len(cs_list) == 0:
        return []
    remain = []
    for cs in cs_list:
        best_i = -1
        best_d = float('inf')
        for i, ds in enumerate(ds_list):
            d = mahalanobis_clusters(cs, ds)
            if d < best_d:
                best_d = d
                best_i = i
        if best_i >= 0 and best_d < threshold:
            ds_list[best_i].merge(cs)
        else:
            remain.append(cs)
    return remain


def count_stats(ds_list, cs_list, rs_ids):
    return (
        sum(c.N for c in ds_list),
        len(cs_list),
        sum(c.N for c in cs_list),
        len(rs_ids),
    )


def bfr(indices, features, n_cluster):
    n_points, dim = features.shape
    threshold = 2.0 * math.sqrt(dim)

    rng = np.random.RandomState(553)
    order = rng.permutation(n_points)
    splits = np.array_split(order, 5)
    chunks = [(indices[idx], features[idx]) for idx in splits if len(idx) > 0]

    intermediate = []

    ids0 = [int(x) for x in chunks[0][0]]
    pts0 = chunks[0][1]
    labels = run_kmeans(pts0, 5 * n_cluster, random_state=553)
    multi, singles = split_by_size(labels, ids0, pts0)

    rs_ids = [pid for pid, _ in singles]
    rs_points = [pt for _, pt in singles]

    rest_ids = []
    rest_pts_list = []
    for ids, pts in multi:
        rest_ids.extend(ids)
        rest_pts_list.append(pts)
    if rest_pts_list:
        rest_pts = np.vstack(rest_pts_list)
    else:
        rest_pts = np.empty((0, dim), dtype=np.float64)
    rest_ids = [int(x) for x in rest_ids]

    ds_list = []
    if rest_pts.shape[0] > 0:
        ds_labels = run_kmeans(rest_pts, n_cluster, random_state=553)
        groups = {}
        for i, lab in enumerate(ds_labels):
            groups.setdefault(int(lab), []).append(i)
        for lab in sorted(groups.keys()):
            idxs = groups[lab]
            ds_list.append(ClusterStat(rest_pts[idxs], [rest_ids[j] for j in idxs]))

    new_cs, rs_ids, rs_points = rs_to_cs_and_rs(rs_ids, rs_points, n_cluster, random_state=553)
    cs_list = new_cs
    intermediate.append(count_stats(ds_list, cs_list, rs_ids))

    for round_i, (ids_c, pts_c) in enumerate(chunks[1:], start=2):
        remain_ids, remain_pts = assign_to_nearest(
            pts_c, [int(x) for x in ids_c], ds_list, threshold
        )
        remain_ids, remain_pts = assign_to_nearest(
            remain_pts, remain_ids, cs_list, threshold
        )
        rs_ids.extend(remain_ids)
        rs_points.extend(remain_pts)

        new_cs, rs_ids, rs_points = rs_to_cs_and_rs(
            rs_ids, rs_points, n_cluster, random_state=553 + round_i
        )
        cs_list.extend(new_cs)
        cs_list = merge_cs_clusters(cs_list, threshold)

        if round_i == len(chunks):
            cs_list = merge_cs_into_ds(ds_list, cs_list, threshold)

        intermediate.append(count_stats(ds_list, cs_list, rs_ids))

    assignment = {}
    for cid, ds in enumerate(ds_list):
        for pid in ds.point_ids:
            assignment[int(pid)] = int(cid)
    for cs in cs_list:
        for pid in cs.point_ids:
            assignment[int(pid)] = -1
    for pid in rs_ids:
        assignment[int(pid)] = -1
    for pid in indices:
        ip = int(pid)
        if ip not in assignment:
            assignment[ip] = -1
    return intermediate, assignment


def write_output(path, intermediate, assignment):
    with open(path, 'w') as f:
        f.write('The intermediate results:\n')
        for i, (ds_pts, cs_n, cs_pts, rs_pts) in enumerate(intermediate, start=1):
            f.write('Round {}: {},{},{},{}\n'.format(i, ds_pts, cs_n, cs_pts, rs_pts))
        f.write('\n')
        f.write('The clustering results:\n')
        for pid in sorted(assignment.keys()):
            f.write('{},{}\n'.format(pid, assignment[pid]))


if __name__ == '__main__':
    input_file = sys.argv[1]
    n_cluster = int(sys.argv[2])
    output_file = sys.argv[3]

    indices, features = load_data(input_file)
    intermediate, assignment = bfr(indices, features, n_cluster)
    write_output(output_file, intermediate, assignment)
