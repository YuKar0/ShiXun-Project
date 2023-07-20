import numpy as np
import matplotlib.pyplot as plt

def predict(X, w, b):
    return np.sign(np.dot(X, w) + b)
def get_data():
    X = np.array([[3, 3], [4, 3], [1, 1]])
    Y = np.array([1, 1, -1])
    return X, Y
def draw_data(X, Y):
    for i in range(len(X)):
        if Y[i] == 1:
            plt.plot(X[i][0], X[i][1], 'ro')
        else:
            plt.plot(X[i][0], X[i][1], 'bx')
def draw_hyperplane(w, b):
    line_x = [0,10]
    line_y = [0,0]
    for i in range(len(line_x)):
        line_y[i] = (-b - w[0] * line_x[i]) / (w[1]+1e-9)
    plt.plot(line_x, line_y)
    plt.show()
def draw(X, Y, w, b):
    draw_data(X, Y)
    draw_hyperplane(w, b)
def init_params():
    w = np.zeros((2,), dtype=np.float)
    b = 0.0
    return w, b
def train(X, Y):
    w, b = init_params()
    delta=1
    for i in range(100):
        choice = -1
        for j in range(len(X)):
            if Y[j] != predict(X[j], w, b):
                choice = j
                break
        if choice == -1:
            break
        w += delta * Y[choice] * X[choice]
        b += delta * Y[choice]
        if i % 10:
            draw(X, Y, w, b)
    return w, b
if __name__ == '__main__':
    X, Y = get_data()
    w, b = train(X, Y)
    print(f"Hyperplane: {w}*X+b={b}=0")
    draw(X, Y, w, b)

#写一个素数判断函数

