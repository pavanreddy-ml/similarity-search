from ise.losses import Loss


class LossRegistry:
    __loss_map = {
        'cosine': Loss,
        'l1': Loss,
        "l2": Loss,
    }

    def __call__(self, loss):
        if isinstance(loss, str):
            if loss in self.__loss_map:
                return self.__loss_map[loss]
            else:
                raise ValueError(f"Unknown loss type: {loss}")
        elif isinstance(loss, Loss):
            return loss
        else:
            raise TypeError(f"loss must be a string or an instance of Loss, got {type(loss)}")