#!/bin/bash
rm -rf docs/
pipenv run pdoc --html psrmodels/
mv html/rlmodels/ docs/ && rm -r html/
rm -r dist/
pipenv run python setup.py sdist
pipenv run twine check dist/* && pipenv run twine upload dist/*