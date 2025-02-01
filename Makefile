# variables set using ?= can be overridden using environment variables
PROJECT_NAME := pyrt
PROJECT_ROOT := .

ifdef PYRT_VENV_ROOT
	PYRT_VENV_PATH ?= $(PYRT_VENV_ROOT)/$(PROJECT_NAME)
else
	PYRT_VENV_PATH ?= $(PROJECT_ROOT)/venv
endif

PYTHON_VERSION := python3
VENV_PYTHON := $(PYRT_VENV_PATH)/bin/python3

venv: setup.cfg
ifdef VIRTUAL_ENV
	@echo "venv already activated: '$(VIRTUAL_ENV)'"
	@exit 1
else
	# include the system site packages using `--system-site-packages`
	$(PYTHON_VERSION) -m venv $(PYRT_VENV_PATH)
	$(VENV_PYTHON) -m pip install -U pip
	$(VENV_PYTHON) -m pip install -e .[quality]
endif

clean:
	rm -rf $(PYRT_VENV_PATH)
	find -iname "*.pyc" -delete
