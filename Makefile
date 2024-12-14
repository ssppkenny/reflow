.EXPORT_ALL_VARIABLES:
OS := macos
ARCH := arm64
SUBDIRS := rlsa djvu_utils utils

all: $(SUBDIRS)
$(SUBDIRS):
	make -C $@

.PHONY: all $(SUBDIRS)

clean:
	rm -rf utils*.so mydjvu*.so rlsa*.so
