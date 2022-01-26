all:
	python main.py

black:
	python -m black --exclude "generated_*" .

clean:
	find . | grep -E "(__pycache__|\.pyc)" | xargs rm -rf
