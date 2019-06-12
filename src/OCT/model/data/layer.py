# This module holds classes representing annotation layers of different types

class _base_layer(object):
    def __init__(self, data, name, type=None, editable=None):
        self._name = name
        self._data = data
        self._type = type
        self._editable = editable
        self._opacity = 100
        self._visible = True

        self._shape = None
        self._position = None
        self.

    # Property getters
    @property
    def name(self):
        return self._name

    @property
    def data(self):
        return self._data

    @property
    def opacity(self):
        return self._opacity

    @property
    def visible(self):
        return self._visible

    @property
    def editable(self):
        return self._editable

    @property
    def shape(self):
        return self._shape

    @property
    def position(self):
        return self._position

    # Property setters
    @property.setter
    def name(self, value):
        if type(value) is not str or len(value) > 20:
            raise ValueError("'name' needs to be a string of length < 20 characters")
        self._name = value

    @property.setter
    def data(self, value):
        if type(value) is not np.ndarray:
            raise ValueError("'data' needs to be of type 'np.ndarray' "
                             "but it is of type {}".format(type(value)))
        self._data = value
        self._shape = self._data.shape

    @property.setter
    def opacity(self, value):
        if value > 100 or value < 0:
            raise ValueError("'opacity' can only have values between 0 and 100."
                             " You tried setting it to {}".format(value))
        self._opacity = value

    @property.setter
    def visible(self, value):
        if value not in [True, False]:
            raise ValueError("'visible' can only be True or False."
                             " You tried setting it to {}".format(value))
        self._visible = value

    @property.setter
    def editable(self, value):
        if value not in [True, False]:
            raise ValueError("'editable' can only be True or False."
                             " You tried setting it to {}".format(value))
        self._editable = value

    @property.setter
    def position(self, value):
        # Tell registered layers that the position changed
        self._position = value

class image_layer(_base_layer):
    """ A class to hold image layers which can not be manipulated by the user.
    For example the original data.

    """
    def __init__(self, name, data, type='image', editable=False):
        super().__init__(name, data, type, editable)

    @property.setter
    def editable(self, value):
        raise ValueError("The 'editable' status of the image layer can not be changed.")

# Line layers
class line_layer(_base_layer):
    def __init__(self, data, name, type='line', editable=True):
        super().__init__(data, name, type, editable)

class rpe_layer(line_layer):
    def __init__(self, data, name='RPE'):
        super().__init__(data, name)

class bm_layer(line_layer):
    def __init__(self, data, name='BM'):
        super().__init__(data, name)

# Area layers
class area_layer(_base_layer):
    def __init__(self, data, name, type='area', editable=True):
        super().__init__(data, name, type, editable)

class drusen_layer(area_layer):
    def __init__(self, data, name='Drusen'):
        super().__init__(data, name)

class hrf_layer(area_layer):
    def __init__(self, data, name='HRF'):
        super().__init__(data, name)

