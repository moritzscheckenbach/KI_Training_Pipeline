import torch.optim.lr_scheduler as lr_scheduler


def get_scheduler(optimizer, config):
    """
    Factory function to create a learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        config: Config dictionary with scheduler parameters

    Returns:
        PyTorch LR Scheduler
    """
    scheduler_type = config.get("scheduler_type", "StepLR")

    if scheduler_type == "StepLR":
        step_size = config.get("scheduler_step_size", 10)
        gamma = config.get("scheduler_gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "MultiStepLR":
        milestones = config.get("scheduler_milestones", [30, 60, 90])
        gamma = config.get("scheduler_gamma", 0.1)
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == "ExponentialLR":
        gamma = config.get("scheduler_gamma", 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == "CosineAnnealingLR":
        T_max = config.get("scheduler_T_max", 50)
        eta_min = config.get("scheduler_eta_min", 0.0001)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == "ReduceLROnPlateau":
        mode = config.get("scheduler_mode", "min")
        factor = config.get("scheduler_factor", 0.1)
        patience = config.get("scheduler_patience", 10)
        min_lr = config.get("scheduler_min_lr", 1e-6)
        return lr_scheduler.ReduceLROnPlateau(optimizer, mode=mode, factor=factor, patience=patience, min_lr=min_lr)

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_scheduler_info():
    """
    Returns information about available schedulers
    """
    return {
        "available_schedulers": ["StepLR", "MultiStepLR", "ExponentialLR", "CosineAnnealingLR", "ReduceLROnPlateau"],
        "description": {
            "StepLR": "Reduces LR every step_size epochs",
            "MultiStepLR": "Reduces LR at specific milestone epochs",
            "ExponentialLR": "Exponential decay every epoch",
            "CosineAnnealingLR": "Cosine annealing schedule",
            "ReduceLROnPlateau": "Reduces LR when metric plateaus",
        },
    }
