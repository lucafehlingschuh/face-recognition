import cv2


class Rescale():
    """
    Rescales image
    """
    def rescale(self, image, x_size=200, y_size=200):
        """
        Rescales the image to the given format.
        :param image: Image as an ndarray
        :param x_size: Default value: 200
        :param y_size: Default value: 200
        :return: Resized image
        """
        return cv2.resize(image, (x_size, y_size))


