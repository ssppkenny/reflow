all: build copy

build:
	python setup.py build_ext --inplace
copy:
	cp rlsa*.so ../

clean:
	rm -rf build; rm rlsa*.so
