
# reinstall and test package
pipenv run pip install -e .
pipenv run pytest test/
# update docs
rm -r docs/
pipenv run pdoc --html psrmodels/
mv html/psrmodels/ docs/ && rm -r html/
