class Component(object):
    # TODO make all components serializable via _getinitargs_
    def __init__(self, params, shape):
        pass

    def initialize(self):
        pass

    def update(self, params, value):
        pass

    def set_tile(self, tile):
        pass

    def get_support_size(self):
        pass

    def get_field(self):
        pass

    def get_param_vector(self):
        return self.params

    def get_param_names(self):
        return self.params

class Parameter(object):
    category = 'pos'
    label = 'pos-1-x'

