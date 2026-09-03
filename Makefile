SHELL := /bin/bash

RESET  := \033[0m
YELLOW := \033[33m
GREEN  := \033[32m
GRAY   := \033[90m
BLUE   := \033[94m
PINK   := \033[95m

# Python / venv
SYS_PY ?= python3
VENV   ?= .venv

# Default: run with the system Python. Pass USE_VENV=1 to use a virtualenv instead.
#   make env             -> install/verify dependencies (system Python)
#   make env USE_VENV=1  -> create .venv and install dependencies there instead
# env is a separate, explicit step: split/train/predict do NOT trigger it
# themselves, so day-to-day runs stay fast. Run `make env` once (or again
# whenever requirements.txt changes).
USE_VENV ?=
ifeq ($(USE_VENV),1)
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
VENV_TARGET := $(PY)
DEPS_STAMP  := $(VENV)/.deps-installed
else
PY          := $(SYS_PY)
PIP         := $(SYS_PY) -m pip
VENV_TARGET :=
DEPS_STAMP  := .deps-installed
endif

# Project dirs
SRC_DIR     ?= srcs
DATASET_DIR ?= datasets
MODEL_DIR   ?= models
OUTPUT_DIR  ?= outputs

# Scripts
SPLIT_PY   := $(SRC_DIR)/split.py
TRAIN_PY   := $(SRC_DIR)/train.py
PREDICT_PY := $(SRC_DIR)/predict.py

# Ensure imports work (e.g., from mlp_manual import ...)
export PYTHONPATH := $(SRC_DIR):$(PYTHONPATH)

# Full pipeline. Assumes `make env` has already been run at least once.
all: dirs split train predict

help:
	@printf "\n$(GREEN)Usage:$(RESET) make [target] [VAR=value]\n\n"
	@printf "  $(GREEN)Setup$(RESET) (run once, or whenever requirements.txt changes):\n"
	@printf "    $(YELLOW)env$(RESET)                install/verify dependencies (system Python)\n"
	@printf "    $(YELLOW)env $(YELLOW)USE_VENV=1$(RESET)     same, but into an isolated .venv\n\n"
	@printf "  $(GREEN)Pipeline$(RESET) (fast, no dependency checks):\n"
	@printf "    $(YELLOW)all$(RESET)                dirs -> split -> train -> predict\n"
	@printf "    $(YELLOW)split$(RESET)              data.csv -> data_train.csv, data_valid.csv\n"
	@printf "    $(YELLOW)train$(RESET)              train the model, save .npz + learning curves\n"
	@printf "    $(YELLOW)predict$(RESET)            evaluate on the valid set, save predictions.csv\n\n"
	@printf "  $(GREEN)Housekeeping$(RESET):\n"
	@printf "    $(YELLOW)clean$(RESET)              remove generated artifacts\n"
	@printf "    $(YELLOW)fclean$(RESET)             remove artifacts + .venv + env stamp\n"
	@printf "    $(YELLOW)re$(RESET)                 clean, then run the full pipeline\n"
	@printf "\n$(GRAY)Typical first run:$(RESET)\n"
	@printf "  make env USE_VENV=1 && make USE_VENV=1\n"
	@printf "\n$(GRAY)Arguments$(RESET) (forwarded to each script; omit them to use the script's defaults):\n"
	@printf "  $(YELLOW)make split SPLIT_ARGS=\"...\"$(RESET)\n"
	@printf "      --valid-ratio R    rows held out for validation, as a fraction (e.g. 0.3)\n"
	@printf "      --seed N           RNG seed for the train/valid shuffle\n"
	@printf "  $(YELLOW)make train TRAIN_ARGS=\"...\"$(RESET)\n"
	@printf "      --layer N [N ...]  hidden-layer sizes (one value per hidden layer)\n"
	@printf "      --epochs N         number of training epochs\n"
	@printf "      --batch_size N     mini-batch size\n"
	@printf "      --learning_rate F  learning rate\n\n"
	@printf "  $(YELLOW)make predict PREDICT_ARGS=\"...\"$(RESET)\n"
	@printf "      --input PATH       CSV file to evaluate (default: the validation split)\n\n"
	@printf "$(GRAY)USAGE:$(RESET)$(BLUE) make split SPLIT_ARGS=\"--valid-ratio 0.2 --seed 43\"\n"
	@printf "       make predict PREDICT_ARGS=\"--input datasets/data_valid.csv\"\n"
	@printf "       make train TRAIN_ARGS=\"--layer 64 32 --epochs 60 --batch_size 32 --learning_rate 0.05\"$(RESET)\n\n"
	@printf "$(GRAY)Full per-script flags:$(RESET) python3 srcs/train.py --help\n"
	@printf "\n"

$(VENV)/bin/python:
	@printf "$(GRAY)[env]$(RESET) creating virtualenv at $(VENV)...\n"
	$(SYS_PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip

env: $(DEPS_STAMP)
	@printf "$(GREEN)[env]$(RESET) dependencies ready ($(if $(filter 1,$(USE_VENV)),$(VENV),system Python)).\n"

$(DEPS_STAMP): requirements.txt | $(VENV_TARGET)
	@printf "$(GRAY)[env]$(RESET) installing requirements...\n"
	$(PIP) install -r requirements.txt
	@touch $(DEPS_STAMP)

dirs:
	@mkdir -p $(DATASET_DIR) $(MODEL_DIR) $(OUTPUT_DIR)

# Optional CLI args passed through to each script (empty -> script defaults).

SPLIT_ARGS   ?=
TRAIN_ARGS   ?=
PREDICT_ARGS ?=

split: $(SPLIT_PY) | dirs
	@printf "$(PINK)[split]$(RESET) running...\n"
	$(PY) $(SPLIT_PY) $(SPLIT_ARGS)

train: $(TRAIN_PY) | dirs
	@printf "$(BLUE)[train]$(RESET) running...\n"
	$(PY) $(TRAIN_PY) $(TRAIN_ARGS)

predict: $(PREDICT_PY) | dirs
	@printf "$(YELLOW)[predict]$(RESET) running...\n"
	$(PY) $(PREDICT_PY) $(PREDICT_ARGS)

clean:
	@printf "$(GREEN)[clean]$(RESET) removing artifacts...\n"
	@rm -f \
		$(MODEL_DIR)/*.npz \
		$(MODEL_DIR)/latest \
		$(MODEL_DIR)/predictions.csv \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

fclean:
	@printf "$(GREEN)[fclean]$(RESET) removing artifacts, venv, and env stamp...\n"
	@rm -rf $(VENV)
	@rm -f .deps-installed
	@rm -f \
		$(MODEL_DIR)/*.npz \
		$(MODEL_DIR)/latest \
		$(MODEL_DIR)/predictions.csv \
		$(OUTPUT_DIR)/*.png \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

re: clean all

.PHONY: all help env dirs split train predict clean fclean re