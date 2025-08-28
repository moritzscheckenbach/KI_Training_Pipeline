from torch.optim import lr_scheduler


def get_scheduler(optimizer, config):
    """
    Factory function to create a learning rate scheduler

    Args:
        optimizer: PyTorch optimizer
        config: Config dictionary with scheduler parameters

    Returns:
        PyTorch LR Scheduler

    Note:
        For ReduceLROnPlateau, use scheduler.step(metric) instead of scheduler.step()
    """
    scheduler_type = config.scheduler.type

    if scheduler_type == "StepLR":
        step_size = config.scheduler.step_size
        gamma = config.scheduler.gamma
        return lr_scheduler.StepLR(
            optimizer,
            step_size=step_size,
            gamma=gamma,
        )

    elif scheduler_type == "MultiStepLR":
        milestones = config.scheduler.milestones
        gamma = config.scheduler.gamma
        return lr_scheduler.MultiStepLR(
            optimizer,
            milestones=milestones,
            gamma=gamma,
        )

    elif scheduler_type == "ExponentialLR":
        gamma = config.scheduler.gamma
        return lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )

    elif scheduler_type == "CosineAnnealingLR":
        T_max = config.scheduler.T_max
        eta_min = config.scheduler.eta_min
        return lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )

    elif scheduler_type == "ReduceLROnPlateau":
        mode = config.scheduler.mode
        factor = config.scheduler.factor
        patience = config.scheduler.patience
        threshold = config.scheduler.threshold
        cooldown = config.scheduler.cooldown
        min_lr = config.scheduler.min_lr

        return lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=mode,
            factor=factor,
            patience=patience,
            threshold=threshold,
            cooldown=cooldown,
            min_lr=min_lr,
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_scheduler_info():
    """
    Returns information about available schedulers and their parameters
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
        "parameters": {
            "common": ["scheduler_type", "scheduler_last_epoch"],
            "StepLR": ["scheduler_step_size", "scheduler_gamma"],
            "MultiStepLR": ["scheduler_milestones", "scheduler_gamma"],
            "ExponentialLR": ["scheduler_gamma"],
            "CosineAnnealingLR": ["scheduler_T_max", "scheduler_eta_min"],
            "ReduceLROnPlateau": [
                "scheduler_mode",
                "scheduler_factor",
                "scheduler_patience",
                "scheduler_threshold",
                "scheduler_threshold_mode",
                "scheduler_cooldown",
                "scheduler_min_lr",
                "scheduler_eps",
                "scheduler_verbose",
            ],
        },
    }
