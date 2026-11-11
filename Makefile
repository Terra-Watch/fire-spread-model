PY ?= .venv/bin/python
test:
	$(PY) -m pytest
check:  # no data needed: every file compiles
	$(PY) -m compileall -q src scripts tests
