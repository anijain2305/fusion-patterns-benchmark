SUBDIRS := $(wildcard */.)

all:
	for dir in $(SUBDIRS); do \
		$(MAKE) -C $$dir; \
	done

black:
	python -m black --exclude "generated_*" .

clean:
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf
