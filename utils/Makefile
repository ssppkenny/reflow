all: build copy

build:
	python setup.py build_ext --inplace

copy:
	cp utils*.so ../  

clean:
	rm -rf build; rm -rf utils*.so
