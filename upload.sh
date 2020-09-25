#!/bin/bash
rm -rf docs/
pipenv run pdoc --html psrmodels/
mv html/psrmodels/ docs/ && rm -rf html/
rm -rf dist/
pipenv run python setup.py sdist
pipenv run twine check dist/* && pipenv run twine upload dist/*