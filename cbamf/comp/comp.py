class ParameterGroup(object):
    category = 'param'

    def __init__(self, params):
        self.p2i = {}
        self.params = np.array(params)

    def _setup_param_dict(self):
        for i, param in enumerate(self.params):
            self.p2i[self.category+'-'+i] = i

    def _update_values(self, params, values):
        for param, value in zip(params, values):
            self.params[self.p2i[param]] = value

    def initialize(self):
        pass

    def update(self, params, values):
        self._update_values(params, values)

    def get_param(self, param):
        return self.params[self.p2i[param]]

    def get_param_vector(self):
        return self.params

    def get_param_names(self):
        pass

    def get_param_categories(self):
        pass

class Component(ParameterGroup):
    # TODO make all components serializable via _getinitargs_
    def __init__(self, params, shape):
        pass

    def set_tile(self, tile):
        pass

    def get_support_size(self, blocks):
        """
        This method returns the actual image area to be modified by
        the update to `blocks'. 

        Parameters:
        -----------
        blocks: list-like, Block object
            an array 
        """
        pass

    def get_pad_size(self):
        pass

    def get_field(self):
        pass

class Prior(ParameterGroup):
    pass
