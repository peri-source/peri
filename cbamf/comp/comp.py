class Component(object):
    # TODO make all components serializable via _getinitargs_
    def __init__(self, params, shape):
        pass

    def update(self, params=None):
        pass

    def get_field(self):
        pass

    def get_params(self):
        return self.params
