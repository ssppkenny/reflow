# How to build
- for Ubuntu run unzip utils/lib/linux/x64/libs.zip -d utils/lib/linux/x64/
- pixi install
- pixi shell
- edit Makefile, set OS and ARCH, for Ubuntu OS=linux, ARCH=x64
- run make
- on Ubuntu run sudo ldconfig $(realpath utils/lib/linux/x64)
- run python main.py (Kivy app)

# To test in a Jupyter notebook
- jupyter lab
- open Testing.ipynb

# Navigation
- Double-click on the right side of a page moves forward
- Double-click on the left side of a page moves backward
- Long mouse click - reflow a page
