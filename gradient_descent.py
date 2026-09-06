w = 5
learning_rate = 0.1

for i in range(10):

    gradient = 2 * w

    w = w - learning_rate * gradient

    print(w)

# θnew​ = θold​ − α∇J(θ)​