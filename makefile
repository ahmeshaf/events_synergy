# Makefile

# Define the default target
.PHONY: all
all: format

# Define the format target
.PHONY: format
format: isort black

# Define the isort target
.PHONY: isort
isort:
	isort .

# Define the black target
.PHONY: black
black:
	black --preview .


