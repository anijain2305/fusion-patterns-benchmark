SUBDIRS := $(wildcard */.)

all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

black:
	python -m black .
