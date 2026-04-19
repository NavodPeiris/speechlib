for building package:

- uv build

for publishing:

- uv publish --token "pypi token"

for install locally for testing:

cd examples
python -m venv venv
venv/scripts/activate

cpu:

- pip install ../dist/speechlib-2.0.0-py3-none-any.whl

gpu:

- pip install ../dist/speechlib-2.0.0-py3-none-any.whl --extra-index-url https://download.pytorch.org/whl/cu126
