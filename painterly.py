# coding:utf-8
import os
import random
import cairo
import numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage
from style import Style, Impressionist, Expressionist, ColoristWash, Pointillist


class Painter:
    def __init__(self, style, output_dir='.'):
        self.style = style
        self.output_dir = output_dir

    def paint(self, src_file):
        # <PIL.Image.Image image mode=RGB size=1925x1280 at 0x10E92F278>
        src_img = Image.open(src_file).convert('RGB')
        src_name = os.path.split(src_file)[1].split('.')[0]

        self.canvas = cairo.ImageSurface(cairo.FORMAT_RGB24, src_img.width, src_img.height)
        self.context = cairo.Context(self.canvas)
        self.context.scale(src_img.width, src_img.height)
        self.context.set_line_cap(cairo.LINE_CAP_ROUND)

        for i, radius in enumerate(self.style.brush_sizes):
            ref_img = src_img.filter(ImageFilter.GaussianBlur(radius=self.style.blur_filter*radius))
            self.paintLayer(ref_img, radius)
            self.canvas.write_to_png(
                os.path.join(self.output_dir, '%s_%s_%d.png' % (self.style.__class__.__name__, src_name, i)))
        return self.canvas

    def paintLayer(self, ref_img, radius):
        S = []
        self.cur_nparray = self.Surface2array(self.canvas)
        self.ref_nparray = self.Image2array(ref_img)
        D = self.img_diff(self.cur_nparray, self.ref_nparray)
        grid = int(self.style.grid_size*radius)

        width = self.canvas.get_width()//grid*grid
        height = self.canvas.get_height()//grid*grid

        ref_l = self.Image2array(ref_img.convert(mode='I'))
        self.gradient_x = ndimage.sobel(ref_l, 0)
        self.gradient_y = ndimage.sobel(ref_l, 1)

        cnt = 0
        for x in range(0, width, grid):
            for y in range(0, height, grid):
                M = D[x:x+grid, y:y+grid]
                areaError = M.sum()/(grid*grid)
                if areaError > self.style.threshold:
                    cnt += 1
                    x1, y1 = np.unravel_index(np.argmax(M), M.shape)
                    s = self.makeSplineStroke(x1+x, y1+y, radius)
                    S.append(s)

        random.shuffle(S)
        print("radius=%d : stroke %d" % (radius, cnt))

        for s in S:
            self.context.set_line_width(max(self.context.device_to_user_distance(2 * radius, 2 * radius)))
            stroke_color = self.ref_nparray[s[0]]/255
            self.context.set_source_rgb(stroke_color[0], stroke_color[1], stroke_color[2])

            self.context.move_to(s[0][0] / self.canvas.get_width(), s[0][1] / self.canvas.get_height())
            for i in range(1, len(s)):
                self.context.line_to(s[i][0] / self.canvas.get_width(), s[i][1] / self.canvas.get_height())
                self.context.move_to(s[i][0] / self.canvas.get_width(), s[i][1] / self.canvas.get_height())

            self.context.close_path()
            self.context.stroke()

    def makeSplineStroke(self, x0, y0, radius):
        K = [(x0, y0)]
        x, y = x0, y0
        lastDx, lastDy = 0, 0
        for i in range(1,self.style.max_stroke_len+1):
            if i > self.style.min_stroke_len and self.color_diff(x0, y0, x, y):
                return K

            gx, gy = self.gradient_x[x, y], self.gradient_y[x, y]
            if (gx*gx+gy*gy)==0:
                return K

            dx, dy = -gy, gx

            if lastDx*dx+lastDy*dy<0:
                dx, dy = -dx, -dy

            dx = self.style.curvature_filter * dx + (1 - self.style.curvature_filter) * lastDx
            dy = self.style.curvature_filter * dy + (1 - self.style.curvature_filter) * lastDy
            dx_norm = dx / np.sqrt(dx * dx + dy * dy)
            dy_norm = dy / np.sqrt(dx * dx + dy * dy)
            x = x + radius * dx_norm
            y = y + radius * dy_norm

            x = int(round(x))
            y = int(round(y))

            if not self.valid_point(x, y):
                return K

            lastDx, lastDy = dx_norm, dy_norm

            K.append((x,y))

        return K

    def img_diff(self, src, ref):
        delta = src - ref
        return np.sqrt(np.sum(delta**2, axis=-1))

    def color_diff(self, x0, y0, x, y):
        stroke_color = self.ref_nparray[x0, y0]
        ref_color = self.ref_nparray[x, y]
        cur_color = self.cur_nparray[x, y]
        return np.linalg.norm(ref_color - cur_color) < np.linalg.norm(ref_color - stroke_color)

    def valid_point(self, x, y):
        return 0<=x<self.canvas.get_width() and 0<=y<self.canvas.get_height()

    def Surface2array(self, surface):
        img = np.array(surface.get_data())
        img = img.reshape(surface.get_height(), surface.get_width(), 4)
        img = np.dstack((img[:, :, 2], img[:, :, 1], img[:, :, 0]))
        img = img.astype(dtype=np.int32)
        return img.swapaxes(0,1)

    def Image2array(self, image):
        img = np.array(image).astype(dtype=np.int32)
        return img.swapaxes(0,1)


if __name__ == '__main__':
    style = Style()
    painter = Painter(style=style, output_dir='test-images/')
    res = painter.paint('test-images/violet.png')

    styles = [Style(), Impressionist(), Expressionist(), ColoristWash(), Pointillist()]
    for s in styles:
        painter = Painter(style=s, output_dir='test-images/')
        res = painter.paint('test-images/BowSnow.jpg')

    styles = [Style(), Impressionist(), Expressionist(), ColoristWash(), Pointillist()]
    for s in styles:
        painter = Painter(style=s, output_dir='test-images/')
        res = painter.paint('test-images/flower.jpg')
