import numpy as np

n, m = map(int, input().split())
input_m = [list(map(int, input().split())) for _ in range(n)]
output_m = [list(map(int, input().split())) for _ in range(m)]

k = n - m + 1

A = []
b = []
for i in range(m):
    for j in range(m):
        row = []
        for di in range(k):
            for dj in range(k):
                row.append(input_m[i + di][j + dj])
        A.append(row)
        b.append(output_m[i][j])

A = np.array(A)
b = np.array(b)

x, _, _, _ = np.linalg.lstsq(A, b, )

kernel = x.reshape((k, k))
for row in kernel:
    print(" ".join(f"{v:.10f}" for v in row))
