SUBDIRS := $(wildcard */.)

all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

black:
	python -m black .

clean:
	rm -rf __pycache__
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir clean; \
	done
