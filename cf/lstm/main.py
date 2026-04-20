import sys
import numpy as np

# ---------- Автодифф-блоки ----------


class Node:
    def __init__(self, op, args):
        self.f_forward = op[0]
        self.f_backward = op[1]
        self.args = args
        self.value = None
        self.saved_args = None
        self.grad = None

    def forward(self):
        if self.value is not None:
            return self.value
        in_vals = [arg.forward() for arg in self.args]
        self.value, self.saved_args = self.f_forward(in_vals)
        return self.value

    def backward(self, grad):
        if self.grad is None:
            self.grad = grad.copy()
        else:
            self.grad += grad
        grads = self.f_backward(grad, self.saved_args)
        for arg, g in zip(self.args, grads):
            arg.backward(g)


def f_var(arr):
    def f(_):
        return arr, ()

    def backward(_, _a):
        return []

    return f, backward


def f_sum(n_args):
    def f(args):
        res = np.zeros_like(args[0])
        for a in args:
            res = res + a
        return res, ()

    def backward(df, _):
        return [df.copy() for _ in range(n_args)]

    return f, backward


def f_had(n_args):
    def f(args):
        res = args[0]
        for a in args[1:]:
            res = res * a
        return res, args

    def backward(df, args):
        out = []
        for i in range(len(args)):
            prod = df.copy()
            for j in range(len(args)):
                if i != j:
                    prod = prod * args[j]
            out.append(prod)
        return out

    return f, backward


def f_mul():
    def f(args):
        a, b = args
        return np.dot(a, b), args

    def backward(df, args):
        a, b = args
        return [np.dot(df, b.T), np.dot(a.T, df)]

    return f, backward


def f_tnh():
    def f(args):
        a = args[0]
        return np.tanh(a), args

    def backward(df, args):
        return [df * (1 / (np.cosh(args[0]) ** 2))]

    return f, backward


def f_sigmoid():
    def f(args):
        a = args[0]
        res = 1.0 / (1.0 + np.exp(-a))
        return res, res

    def backward(df, res):
        return [df * res * (1 - res)]

    return f, backward


# --------- Вспомогательные функции -----------


def print_vector(v):
    print(" ".join(f"{x:.10f}" for x in v))


def print_matrix(m):
    for r in m:
        print(" ".join(f"{x:.10f}" for x in r))


def read_vector_n(fn, N):
    return list(map(float, fn().split()))


def read_matrix_n(fn, N):
    return [list(map(float, fn().split())) for _ in range(N)]


# ---------- Структура входных параметров ----------


class LSTMParams:
    def __init__(
        self,
        N,
        W_f,
        U_f,
        B_f,
        W_i,
        U_i,
        B_i,
        W_o,
        U_o,
        B_o,
        W_c,
        U_c,
        B_c,
        M,
        h0,
        c0,
        X,
        o_grads,
    ):
        self.N = N
        self.W_f = W_f
        self.U_f = U_f
        self.B_f = B_f
        self.W_i = W_i
        self.U_i = U_i
        self.B_i = B_i
        self.W_o = W_o
        self.U_o = U_o
        self.B_o = B_o
        self.W_c = W_c
        self.U_c = U_c
        self.B_c = B_c
        self.M = M
        self.h0 = h0
        self.c0 = c0
        self.X = X
        self.o_grads = o_grads

    @classmethod
    def from_input(cls):
        fn = input
        N = int(fn())
        W_f = read_matrix_n(fn, N)
        U_f = read_matrix_n(fn, N)
        B_f = read_vector_n(fn, N)
        W_i = read_matrix_n(fn, N)
        U_i = read_matrix_n(fn, N)
        B_i = read_vector_n(fn, N)
        W_o = read_matrix_n(fn, N)
        U_o = read_matrix_n(fn, N)
        B_o = read_vector_n(fn, N)
        W_c = read_matrix_n(fn, N)
        U_c = read_matrix_n(fn, N)
        B_c = read_vector_n(fn, N)
        M = int(fn())
        h0 = read_vector_n(fn, N)
        c0 = read_vector_n(fn, N)
        X = [read_vector_n(fn, N) for _ in range(M)]
        o_grads = [read_vector_n(fn, N) for _ in range(M)]
        return cls(
            N,
            W_f,
            U_f,
            B_f,
            W_i,
            U_i,
            B_i,
            W_o,
            U_o,
            B_o,
            W_c,
            U_c,
            B_c,
            M,
            h0,
            c0,
            X,
            o_grads,
        )


# ---------- Построение LSTM-вычисленного графа ----------


def build_lstm_graph(params):
    N = params.N
    # Оборачиваем параметры как Node
    W_f = Node(f_var(np.array(params.W_f)), [])
    U_f = Node(f_var(np.array(params.U_f)), [])
    B_f = Node(f_var(np.array(params.B_f)), [])

    W_i = Node(f_var(np.array(params.W_i)), [])
    U_i = Node(f_var(np.array(params.U_i)), [])
    B_i = Node(f_var(np.array(params.B_i)), [])

    W_o = Node(f_var(np.array(params.W_o)), [])
    U_o = Node(f_var(np.array(params.U_o)), [])
    B_o = Node(f_var(np.array(params.B_o)), [])

    W_c = Node(f_var(np.array(params.W_c)), [])
    U_c = Node(f_var(np.array(params.U_c)), [])
    B_c = Node(f_var(np.array(params.B_c)), [])

    param_nodes = [W_f, U_f, B_f, W_i, U_i, B_i, W_o, U_o, B_o, W_c, U_c, B_c]

    # Исходные состояния (делаем как Node, чтобы потом взять grad)
    h0_node = Node(f_var(np.array(params.h0)), [])
    c0_node = Node(f_var(np.array(params.c0)), [])
    # списки входов
    X_nodes = [Node(f_var(np.array(x)), []) for x in params.X]

    h_nodes = []
    c_nodes = []
    ht_prev = h0_node
    ct_prev = c0_node

    for t in range(params.M):
        xt = X_nodes[t]

        # f_t = sigmoid(W_f x_t + U_f h_{t-1} + B_f)
        Wx = Node(f_mul(), [W_f, xt])
        Uh = Node(f_mul(), [U_f, ht_prev])
        s1 = Node(f_sum(3), [Wx, Uh, B_f])
        f_t = Node(f_sigmoid(), [s1])

        # i_t = sigmoid(W_i x_t + U_i h_{t-1} + B_i)
        Wx = Node(f_mul(), [W_i, xt])
        Uh = Node(f_mul(), [U_i, ht_prev])
        s2 = Node(f_sum(3), [Wx, Uh, B_i])
        i_t = Node(f_sigmoid(), [s2])

        # o_t = sigmoid(W_o x_t + U_o h_{t-1} + B_o)
        Wx = Node(f_mul(), [W_o, xt])
        Uh = Node(f_mul(), [U_o, ht_prev])
        s3 = Node(f_sum(3), [Wx, Uh, B_o])
        o_t = Node(f_sigmoid(), [s3])

        # c~ = tanh(W_c x_t + U_c h_{t-1} + B_c)
        Wx = Node(f_mul(), [W_c, xt])
        Uh = Node(f_mul(), [U_c, ht_prev])
        s4 = Node(f_sum(3), [Wx, Uh, B_c])
        c_bar = Node(f_tnh(), [s4])

        # c_t = f_t ∘ c_{t-1} + i_t ∘ c_bar
        fc = Node(f_had(2), [f_t, ct_prev])
        ic = Node(f_had(2), [i_t, c_bar])
        ct = Node(f_sum(2), [fc, ic])

        # h_t = o_t ∘ tanh(c_t)
        tnh_ct = Node(f_tnh(), [ct])
        ht = Node(f_had(2), [o_t, tnh_ct])

        # Добавляем в списки состояний
        h_nodes.append(ht)
        c_nodes.append(ct)

        ht_prev = ht
        ct_prev = ct

    return {
        "h_nodes": h_nodes,
        "c_nodes": c_nodes,
        "X_nodes": X_nodes,
        "param_nodes": param_nodes,
        "h0_node": h0_node,
        "c0_node": c0_node,
    }


# ---------- Основной solver ----------


def main():
    params = LSTMParams.from_input()
    G = build_lstm_graph(params)

    h_nodes = G["h_nodes"]
    c_nodes = G["c_nodes"]
    X_nodes = G["X_nodes"]
    param_nodes = G["param_nodes"]
    h0_node = G["h0_node"]
    c0_node = G["c0_node"]

    # Прямой проход
    for node in h_nodes:
        node.forward()

    # Обратный проход (град по выходам - в обратном порядке!)
    for t in range(params.M):
        h_nodes[params.M - 1 - t].backward(np.array(params.o_grads[t]))

    # ------ ВЫВОД ---------

    # M векторов выходов сети o_t (= h_t)
    for node in h_nodes:
        print_vector(node.value)

    # h_M и c_M (последние состояния)
    print_vector(h_nodes[-1].value)
    print_vector(c_nodes[-1].value)

    # M векторов производных по входам x_t в ОБРАТНОМ ПОРЯДКЕ
    for node in reversed(X_nodes):
        grad = node.grad if node.grad is not None else np.zeros(params.N)
        print_vector(grad)

    # Два вектора производных по h0 и c0
    grad_h0 = h0_node.grad if h0_node.grad is not None else np.zeros(params.N)
    grad_c0 = c0_node.grad if c0_node.grad is not None else np.zeros(params.N)
    print_vector(grad_h0)
    print_vector(grad_c0)

    # Градиенты по всем параметрам, в нужном порядке
    for param in param_nodes:
        # Если параметр - матрица, выводим как матрицу, если вектор - как вектор
        grad = param.grad if param.grad is not None else np.zeros_like(param.value)
        if isinstance(grad, np.float64):
            print(grad)
        elif grad.ndim == 2:
            print_matrix(grad)
        else:
            print_vector(grad)


if __name__ == "__main__":
    main()
