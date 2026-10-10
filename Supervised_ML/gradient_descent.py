w = 5
# learning_rate = 0.1
# learning_rate = 0.01
# learning_rate = 0.5
learning_rate = 1.1

for i in range(10):
    loss = w ** 2
    gradient = 2 * w
    w = w - learning_rate * gradient
    # print(w)
    print(f"Iteration: {i + 1}, w: {w}, Loss: {loss}")

# θnew​ = θold​ − α∇J(θ)​