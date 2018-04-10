class Style:
    def __init__(self):
        self.threshold = 10
        self.brush_sizes = [64, 8, 4, 2]
        self.curvature_filter = 1.
        self.blur_filter = .5
        self.min_stroke_len = 4
        self.max_stroke_len = 16
        self.grid_size = 1.


class Impressionist(Style):
    def __init__(self):
        super().__init__()
        self.threshold = 100
        self.brush_sizes = [8, 4, 2]
        self.curvature_filter = 1.
        self.blur_filter = .5
        self.min_stroke_len = 4
        self.max_stroke_len = 16
        self.grid_size = 1.


class Expressionist(Style):
    def __init__(self):
        super().__init__()
        self.threshold = 50
        self.brush_sizes = [8, 4, 2]
        self.curvature_filter = .25
        self.blur_filter = .5
        self.min_stroke_len = 10
        self.max_stroke_len = 16
        self.grid_size = 1.


class ColoristWash(Style):
    def __init__(self):
        super().__init__()
        self.threshold = 200
        self.brush_sizes = [8, 4, 2]
        self.curvature_filter = 1.
        self.blur_filter = .5
        self.min_stroke_len = 4
        self.max_stroke_len = 16
        self.grid_size = 1.


class Pointillist(Style):
    def __init__(self):
        super().__init__()
        self.threshold = 100
        self.brush_sizes = [4, 2]
        self.curvature_filter = 1.
        self.blur_filter = .5
        self.min_stroke_len = 0
        self.max_stroke_len = 0
        self.grid_size = 0.5
