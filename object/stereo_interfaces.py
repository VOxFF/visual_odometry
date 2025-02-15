from abc import ABC, abstractmethod

class StereoParamsInterface(ABC):
    """ Abstract interface for stereo calibration parameters. """

    @abstractmethod
    def load_params(self, input_data):
        """ Load stereo calibration parameters from YAML file or string. """
        pass

    @abstractmethod
    def get_intrinsics(self):
        """ Return camera intrinsic matrices. """
        pass

    @abstractmethod
    def get_baseline(self):
        """ Return the stereo camera baseline. """
        pass


class StereoRectificationInterface(ABC):
    """ Abstract interface for stereo rectification. """

    @abstractmethod
    def rectify_images(self, img_left, img_right):
        """ Rectify stereo images using calibration parameters. """
        pass

    @abstractmethod
    def get_rectification_matrices(self):
        """ Return rectification matrices. """
        pass
