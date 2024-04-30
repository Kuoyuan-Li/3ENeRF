models = {}


def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, config):
    model = models[name](config) # config: model config
    return model


from . import nerf, neus, rignerf, geometry, texture, deformation, rignerf_texture
