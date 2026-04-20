import numpy as np


class LSTMNetwork:
    def __init__(self):
        self.N = None  # Размер векторов LSTM
        self.M = None  # Число элементов последовательности

        # Матрицы и векторы параметров
        self.W_f, self.U_f, self.B_f = None, None, None
        self.W_i, self.U_i, self.B_i = None, None, None
        self.W_o, self.U_o, self.B_o = None, None, None
        self.W_c, self.U_c, self.B_c = None, None, None

        # Входные данные
        self.h_0 = None  # Начальный вектор состояния
        self.c_0 = None  # Начальная память
        self.x_t = []  # Входные векторы

        # Выходные данные
        self.o_t = []  # Выходы сети
        self.h_M = None  # Последний вектор состояния
        self.c_M = None  # Последняя память
        self.dx_t = []  # Производные по входам
        self.dh_0 = None  # Производная по начальному состоянию
        self.dc_0 = None  # Производная по начальной памяти

        # Градиенты параметров
        self.dW_f, self.dU_f, self.dB_f = None, None, None
        self.dW_i, self.dU_i, self.dB_i = None, None, None
        self.dW_o, self.dU_o, self.dB_o = None, None, None
        self.dW_c, self.dU_c, self.dB_c = None, None, None

    def sigmoid(self, x):
        """Сигмоидная функция активации"""
        return 1 / (1 + np.exp(-np.clip(x, -500, 500)))

    def tanh(self, x):
        """Гиперболический тангенс"""
        return np.tanh(x)

    def read_input(self, input_data):
        """Чтение входных данных"""
        lines = input_data.strip().split("\n")

        # Чтение размера векторов N
        self.N = int(lines[0])

        line_idx = 1

        # Чтение матриц и векторов параметров
        # W_f
        self.W_f = []
        for i in range(self.N):
            self.W_f.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.W_f = np.array(self.W_f)

        # U_f
        self.U_f = []
        for i in range(self.N):
            self.U_f.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.U_f = np.array(self.U_f)

        # B_f
        self.B_f = np.array(list(map(float, lines[line_idx].split())))
        line_idx += 1

        # W_i
        self.W_i = []
        for i in range(self.N):
            self.W_i.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.W_i = np.array(self.W_i)

        # U_i
        self.U_i = []
        for i in range(self.N):
            self.U_i.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.U_i = np.array(self.U_i)

        # B_i
        self.B_i = np.array(list(map(float, lines[line_idx].split())))
        line_idx += 1

        # W_o
        self.W_o = []
        for i in range(self.N):
            self.W_o.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.W_o = np.array(self.W_o)

        # U_o
        self.U_o = []
        for i in range(self.N):
            self.U_o.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.U_o = np.array(self.U_o)

        # B_o
        self.B_o = np.array(list(map(float, lines[line_idx].split())))
        line_idx += 1

        # W_c
        self.W_c = []
        for i in range(self.N):
            self.W_c.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.W_c = np.array(self.W_c)

        # U_c
        self.U_c = []
        for i in range(self.N):
            self.U_c.append(list(map(float, lines[line_idx].split())))
            line_idx += 1
        self.U_c = np.array(self.U_c)

        # B_c
        self.B_c = np.array(list(map(float, lines[line_idx].split())))
        line_idx += 1

        # Чтение M
        self.M = int(lines[line_idx])
        line_idx += 1

        # Чтение h_0 и c_0
        self.h_0 = np.array(list(map(float, lines[line_idx].split())))
        line_idx += 1
        self.c_0 = np.array(list(map(float, lines[line_idx].split())))
        line_idx += 1

        # Чтение входных векторов x_t
        self.x_t = []
        for i in range(self.M):
            self.x_t.append(np.array(list(map(float, lines[line_idx].split()))))
            line_idx += 1

        # Чтение производных по выходам в обратном порядке
        self.do_t = []
        for i in range(self.M):
            self.do_t.append(np.array(list(map(float, lines[line_idx].split()))))
            line_idx += 1
        self.do_t = self.do_t[::-1]  # Переворачиваем в правильный порядок

    def forward_pass(self):
        """Прямой проход через LSTM"""
        h_prev = self.h_0
        c_prev = self.c_0

        self.hidden_states = [h_prev]
        self.cell_states = [c_prev]
        self.o_t = []

        # Сохраняем промежуточные значения для обратного прохода
        self.f_gates = []
        self.i_gates = []
        self.o_gates = []
        self.c_tildes = []

        for t in range(self.M):
            x = self.x_t[t]

            # Forget gate
            f_t = self.sigmoid(self.W_f @ x + self.U_f @ h_prev + self.B_f)

            # Input gate
            i_t = self.sigmoid(self.W_i @ x + self.U_i @ h_prev + self.B_i)

            # Candidate values
            c_tilde = self.tanh(self.W_c @ x + self.U_c @ h_prev + self.B_c)

            # Cell state
            c_t = f_t * c_prev + i_t * c_tilde

            # Output gate
            o_t = self.sigmoid(self.W_o @ x + self.U_o @ h_prev + self.B_o)

            # Hidden state
            h_t = o_t * self.tanh(c_t)

            # Сохраняем значения
            self.f_gates.append(f_t)
            self.i_gates.append(i_t)
            self.o_gates.append(o_t)
            self.c_tildes.append(c_tilde)
            self.hidden_states.append(h_t)
            self.cell_states.append(c_t)
            self.o_t.append(h_t)

            # Обновляем предыдущие состояния
            h_prev = h_t
            c_prev = c_t

        self.h_M = h_prev
        self.c_M = c_prev

    def backward_pass(self):
        """Обратный проход для вычисления градиентов"""
        # Инициализация градиентов
        self.dW_f = np.zeros_like(self.W_f)
        self.dU_f = np.zeros_like(self.U_f)
        self.dB_f = np.zeros_like(self.B_f)

        self.dW_i = np.zeros_like(self.W_i)
        self.dU_i = np.zeros_like(self.U_i)
        self.dB_i = np.zeros_like(self.B_i)

        self.dW_o = np.zeros_like(self.W_o)
        self.dU_o = np.zeros_like(self.U_o)
        self.dB_o = np.zeros_like(self.B_o)

        self.dW_c = np.zeros_like(self.W_c)
        self.dU_c = np.zeros_like(self.U_c)
        self.dB_c = np.zeros_like(self.B_c)

        self.dx_t = []

        # Начальные градиенты
        dh_next = np.zeros(self.N)
        dc_next = np.zeros(self.N)

        # Обратный проход по времени
        for t in reversed(range(self.M)):
            # Градиент от выхода
            dh = self.do_t[t] + dh_next

            # Градиенты через output gate
            do = dh * self.tanh(self.cell_states[t + 1])
            do_input = do * self.o_gates[t] * (1 - self.o_gates[t])

            # Градиент по cell state
            dc = (
                dh * self.o_gates[t] * (1 - self.tanh(self.cell_states[t + 1]) ** 2)
                + dc_next
            )

            # Градиенты через forget gate
            df = dc * self.cell_states[t]
            df_input = df * self.f_gates[t] * (1 - self.f_gates[t])

            # Градиенты через input gate
            di = dc * self.c_tildes[t]
            di_input = di * self.i_gates[t] * (1 - self.i_gates[t])

            # Градиенты через candidate
            dc_tilde = dc * self.i_gates[t]
            dc_tilde_input = dc_tilde * (1 - self.c_tildes[t] ** 2)

            # Накопление градиентов параметров
            x = self.x_t[t]
            h_prev = self.hidden_states[t]

            self.dW_f += np.outer(df_input, x)
            self.dU_f += np.outer(df_input, h_prev)
            self.dB_f += df_input

            self.dW_i += np.outer(di_input, x)
            self.dU_i += np.outer(di_input, h_prev)
            self.dB_i += di_input

            self.dW_o += np.outer(do_input, x)
            self.dU_o += np.outer(do_input, h_prev)
            self.dB_o += do_input

            self.dW_c += np.outer(dc_tilde_input, x)
            self.dU_c += np.outer(dc_tilde_input, h_prev)
            self.dB_c += dc_tilde_input

            # Градиент по входу
            dx = (
                self.W_f.T @ df_input
                + self.W_i.T @ di_input
                + self.W_o.T @ do_input
                + self.W_c.T @ dc_tilde_input
            )
            self.dx_t.append(dx)

            # Градиенты для следующего шага
            dh_next = (
                self.U_f.T @ df_input
                + self.U_i.T @ di_input
                + self.U_o.T @ do_input
                + self.U_c.T @ dc_tilde_input
            )
            dc_next = dc * self.f_gates[t]

        # Градиенты по начальным состояниям
        self.dh_0 = dh_next
        self.dc_0 = dc_next

        # Переворачиваем dx_t в правильный порядок
        self.dx_t = self.dx_t[::-1]

    def format_output(self):
        """Форматирование выходных данных"""
        output = []

        # Выходы сети
        for o in self.o_t:
            output.append(" ".join(f"{val:.16E}" for val in o))

        # Последние состояния
        output.append(" ".join(f"{val:.16E}" for val in self.h_M))
        output.append(" ".join(f"{val:.16E}" for val in self.c_M))

        # Производные по входам в обратном порядке
        for dx in reversed(self.dx_t):
            output.append(" ".join(f"{val:.16E}" for val in dx))

        # Производные по начальным состояниям
        output.append(" ".join(f"{val:.16E}" for val in self.dh_0))
        output.append(" ".join(f"{val:.16E}" for val in self.dc_0))

        # Градиенты параметров
        # dW_f
        for row in self.dW_f:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dU_f
        for row in self.dU_f:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dB_f
        output.append(" ".join(f"{val:.16E}" for val in self.dB_f))

        # dW_i
        for row in self.dW_i:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dU_i
        for row in self.dU_i:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dB_i
        output.append(" ".join(f"{val:.16E}" for val in self.dB_i))

        # dW_o
        for row in self.dW_o:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dU_o
        for row in self.dU_o:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dB_o
        output.append(" ".join(f"{val:.16E}" for val in self.dB_o))

        # dW_c
        for row in self.dW_c:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dU_c
        for row in self.dU_c:
            output.append(" ".join(f"{val:.16E}" for val in row))
        # dB_c
        output.append(" ".join(f"{val:.16E}" for val in self.dB_c))

        return "\n".join(output)


# Пример использования
if __name__ == "__main__":
    # Пример входных данных из задачи
    input_data = """1
-3
2
1
1
-2
-2
-3
-1
-2
1
-2
-1
1
1
-3
2
1
-1
1"""

    lstm = LSTMNetwork()
    lstm.read_input(input_data)
    lstm.forward_pass()
    lstm.backward_pass()

    print("Результат работы LSTM сети:")
    print(lstm.format_output())
