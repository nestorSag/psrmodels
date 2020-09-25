
# reinstall and test package
pipenv run pytest test/ && pipenv run pip install -e .
# update docs
rm -rf docs/
pipenv run pdoc --html psrmodels/
mv html/psrmodels/ docs/ && rm -r html/
