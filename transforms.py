from PIL.Image import Image


class Gray2rgb(object):
    def __call__(self, pic: Image):
        return pic.convert('RGB')

    def __repr__(self):
        return self.__class__.__name__ + '()'
