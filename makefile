# Makefile for code quality checks and testing

# Define the Python interpreter
PYTHON := python

# Define the commands for code quality tools
ISORT := isort .
BLACK := black --preview .

# Define targets
.PHONY: isort black format 

# Target to run isort
isort:
	$(ISORT)

# Target to run black
black:
	$(BLACK)

format: isort black 
