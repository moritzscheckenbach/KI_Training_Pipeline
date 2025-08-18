import torch.optim as optim


def get_optimizer(model_parameters, config):
    """
    Factory function to create a PyTorch optimizer

    Args:
        model_parameters: Model parameters (from model.parameters())
        config: Config dictionary with optimizer parameters

    Returns:
        PyTorch Optimizer
    """
    optimizer_type = config.get("optimizer_type", "Adam")
    learning_rate = config.get("learning_rate", 0.001)

    if optimizer_type == "Adam":
        weight_decay = config.get("optimizer_weight_decay", 0.0001)
        betas = config.get("optimizer_betas", (0.9, 0.999))
        eps = config.get("optimizer_eps", 1e-8)
        amsgrad = config.get("optimizer_amsgrad", False)

        return optim.Adam(model_parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    elif optimizer_type == "AdamW":
        weight_decay = config.get("optimizer_weight_decay", 0.01)
        betas = config.get("optimizer_betas", (0.9, 0.999))
        eps = config.get("optimizer_eps", 1e-8)
        amsgrad = config.get("optimizer_amsgrad", False)

        return optim.AdamW(model_parameters, lr=learning_rate, betas=betas, eps=eps, weight_decay=weight_decay, amsgrad=amsgrad)

    elif optimizer_type == "SGD":
        momentum = config.get("optimizer_momentum", 0.9)
        weight_decay = config.get("optimizer_weight_decay", 0.0001)
        dampening = config.get("optimizer_dampening", 0)
        nesterov = config.get("optimizer_nesterov", False)

        return optim.SGD(model_parameters, lr=learning_rate, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)

    elif optimizer_type == "RMSprop":
        alpha = config.get("optimizer_alpha", 0.99)
        eps = config.get("optimizer_eps", 1e-8)
        weight_decay = config.get("optimizer_weight_decay", 0)
        momentum = config.get("optimizer_momentum", 0)
        centered = config.get("optimizer_centered", False)

        return optim.RMSprop(model_parameters, lr=learning_rate, alpha=alpha, eps=eps, weight_decay=weight_decay, momentum=momentum, centered=centered)

    elif optimizer_type == "Adagrad":
        lr_decay = config.get("optimizer_lr_decay", 0)
        weight_decay = config.get("optimizer_weight_decay", 0)
        initial_accumulator_value = config.get("optimizer_initial_accumulator_value", 0)
        eps = config.get("optimizer_eps", 1e-10)

        return optim.Adagrad(model_parameters, lr=learning_rate, lr_decay=lr_decay, weight_decay=weight_decay, initial_accumulator_value=initial_accumulator_value, eps=eps)

    elif optimizer_type == "Adadelta":
        rho = config.get("optimizer_rho", 0.9)
        eps = config.get("optimizer_eps", 1e-6)
        weight_decay = config.get("optimizer_weight_decay", 0)

        return optim.Adadelta(model_parameters, lr=learning_rate, rho=rho, eps=eps, weight_decay=weight_decay)

    else:
        raise ValueError(f"Unknown optimizer type: {optimizer_type}")


def get_optimizer_info():
    """
    Returns information about available optimizers
    """
    return {
        "available_optimizers": ["Adam", "AdamW", "SGD", "RMSprop", "Adagrad", "Adadelta"],
        "description": {
            "Adam": "Adaptive Moment Estimation - most popular choice",
            "AdamW": "Adam with decoupled weight decay - better for transformers",
            "SGD": "Stochastic Gradient Descent with momentum - classic choice",
            "RMSprop": "Root Mean Square Propagation - good for RNNs",
            "Adagrad": "Adaptive Gradient - good for sparse data",
            "Adadelta": "Extension of Adagrad - no learning rate needed",
        },
        "recommendations": {"general": "Adam", "transformers": "AdamW", "computer_vision": "SGD with momentum", "rnn/lstm": "RMSprop"},
    }
