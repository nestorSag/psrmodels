
# reinstall and test package
pipenv run pip install -e . && pipenv run pytest test/
# update docs
rm -rf docs/
pipenv run pdoc --html psrmodels/
mv html/psrmodels/ docs/ && rm -r html/
