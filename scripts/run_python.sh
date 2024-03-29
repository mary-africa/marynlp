# Run python related scripts

# Installing the dependencies
# ------------------------------------
cd python

# Installing base dependencies
python -m pip install -r requirements.txt

# installing dependencies for development / testing 
python -m pip install -r requirements-dev.txt

# installing the package
python setup.py install

# code linting
$SHELL scripts/run_linter.sh

# execute the test file
$SHELL scripts/run_tests.sh
