from numpy import insert
from kivy.app import App
from kivy.uix.floatlayout import FloatLayout
from kivy.factory import Factory
from kivy.properties import ObjectProperty
from kivy.uix.popup import Popup
from kivy.logger import Logger
from kivy.properties import ListProperty, NumericProperty

# import rlsafast
from kivy.uix.behaviors import ButtonBehavior
from kivymd.uix.behaviors import TouchBehavior

import kivy
import utils
from io import BytesIO
from kivy import platform
from urllib.parse import unquote

from kivy.core.window import Window
from kivy.uix.gridlayout import GridLayout
import kivy.uix.image
from PIL import Image, ImageOps
from kivy.clock import Clock
from kivy.core.image import Image as CoreImage

import mydjvulib
import cv2
import reflow

files_path = ""


class LoadDialog(FloatLayout):
    load = ObjectProperty(None)
    cancel = ObjectProperty(None)


class Root(FloatLayout, TouchBehavior):
    loadfile = ObjectProperty(None)
    text_input = ObjectProperty(None)
    image = ObjectProperty(None)
    pageno = 0
    scheduled_event = None
    filename = None
    reflowed = False
    count = 0
    # duration_long_touch = NumericProperty(1.0)
    locked = False

    def dismiss_popup(self):
        self._popup.dismiss()

    def handle_selection(self, selection):
        print("SELECTION = ")
        print(selection)
        self.load(selection, selection)

    def show_load(self):
        content = LoadDialog(load=self.load, cancel=self.dismiss_popup)
        self._popup = Popup(title="Load file", content=content, size_hint=(0.9, 0.9))
        self._popup.open()

    def add(self):
        pass

    def load(self, path, filename):
        self.pageno = 0
        self.filename = filename[0]
        page_width = 1.2 * Window.width
        if filename[0].endswith(".djvu"):
            img = self.load_djvu(self.pageno, self.filename)
        else:
            img = self.load_pdf(self.pageno, self.filename, page_width)
        data = BytesIO()
        img.save(data, format="png")
        data.seek(0)
        im = CoreImage(BytesIO(data.read()), ext="png")
        self.ids.image.texture = im.texture
        self.dismiss_popup()

    def on_long_touch(self, touch, *args):
        if touch.is_double_tap:
            return
        if not self.locked:
            self.locked = True
            self.reflowed = not self.reflowed
            try:
                self.update(False)
            except Exception as e:
                print(e)
        self.locked = False

    def on_double_tap(self, touch):
        super(Root, self).on_double_tap(touch)
        if not self.locked:
            self.locked = True
            width, height = Window.size
            if self.filename is None:
                return
            self.mouse_x, _ = touch.pos
            # self.mouse_x, _ = touch.pos
            x, _ = touch.pos

            if x > width / 2:
                self.pageno += 1
                self.update(True)
            else:
                self.pageno -= 1 if self.pageno > 0 else 0
                self.update(True)
        self.locked = False

    def load_djvu(self, pageno, filepath):
        arr = mydjvulib.get_image_as_arrray(pageno, filepath)
        _, bw = cv2.threshold(arr, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        return Image.fromarray(bw, "L")

    def load_pdf(self, pageno, filepath, page_width):
        b = utils.get_page_for_display(pageno, filepath, page_width)
        w, h = utils.get_page_size_for_display(pageno, filepath, page_width)
        return Image.frombytes("RGBA", (w, h), b)

    def update(self, load_new_page):
        if self.filename is None:
            return
        page_width = 1.2 * Window.width
        if load_new_page:
            if self.filename.endswith(".djvu"):
                self.img = self.load_djvu(self.pageno, self.filename)
            else:
                self.img = self.load_pdf(self.pageno, self.filename, page_width)
        if self.reflowed:
            new_image = reflow.reflow(self.img)
            data = BytesIO()
            new_image.save(data, format="png")
            data.seek(0)
            im = CoreImage(BytesIO(data.read()), ext="png")
            self.ids.image.texture = im.texture
        else:
            data = BytesIO()
            self.img.save(data, format="png")
            data.seek(0)
            im = CoreImage(BytesIO(data.read()), ext="png")
            self.ids.image.texture = im.texture


class Editor(App):
    pass


Factory.register("Root", cls=Root)
Factory.register("LoadDialog", cls=LoadDialog)


if __name__ == "__main__":
    app = Editor()
    app.run()
