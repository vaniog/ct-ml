M = int(input())
table = [int(input()) for _ in range(2**M)]

minterms = [i for i, v in enumerate(table) if v == 1]
D = 2
n = len(minterms)

print(D)
print(n, 1)

for i in minterms:
    bits = [(i >> (M - 1 - j)) & 1 for j in range(M)]
    weights = []
    bias = 0.0
    for bit in bits:
        if bit == 1:
            weights.append(1.0)
            bias -= 1.0
        else:
            weights.append(-1.0)
    bias += 0.5
    print(" ".join(f"{w:.10g}" for w in weights), f"{bias:.10g}")

if n == 0:
    print("-0.5")
else:
    print(" ".join(["1.0"] * n) + " -0.5")
