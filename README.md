# Introduction
This repo contains my experiments making models that employ dynamic residual scaling. These residual coefficients can come from auxiliary networks or can simply be learnable parameters. 

# Methods
The task I chose is CIFAR classification using a CNN, but I have experimented with this idea on pure resnets as well as the residual connections in transformer blocks. In this instance I created two models that both use residual connections. One of them using standard residual connections and the other uses a coefficient network that takes the residual as input and produces coefficients that are used to scale it before it is added to the output of the next layer. 

The following would be a typical residual connection (where $$x$$ is the original input):

$$
y = f(x) + x
$$

This would be an example of the dynamic residual scaling where $$g(x)$$ is the coefficient network:

$$
y = f(x) + (x * g(x))
$$

# Visualizations
## CNN
![CNN](https://github.com/NoInitRD/Dynamic-Residuals/blob/main/figureDynConv.png)

## Residual Network
![Residual Network](https://github.com/NoInitRD/Dynamic-Residuals/blob/main/figureDynRes.png)
