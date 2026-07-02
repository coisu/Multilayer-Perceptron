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
#   make            -> run with the system Python
#   make USE_VENV=1 -> create .venv, install requirements, then run
USE_VENV ?=
ifeq ($(USE_VENV),1)
PY          := $(VENV)/bin/python
PIP         := $(VENV)/bin/pip
VENV_TARGET := $(PY)
PREP        := deps
else
PY          := $(SYS_PY)
PIP         := $(SYS_PY) -m pip
VENV_TARGET :=
PREP        :=
endif

# Project dirs
SRC_DIR     ?= srcs
DATASET_DIR ?= datasets
MODEL_DIR   ?= models
OUTPUT_DIR  ?= outputs

# Scripts
SPLIT_PY    := $(SRC_DIR)/split.py
TRAIN_PY    := $(SRC_DIR)/train.py
PREDICT_PY  := $(SRC_DIR)/predict.py

# Ensure imports work (e.g., from mlp_manual import ...)
export PYTHONPATH := $(SRC_DIR):$(PYTHONPATH)

# Default pipeline (system Python by default; USE_VENV=1 to run inside .venv)
all: $(PREP) dirs split train predict

help:
	@printf "\n$(GREEN)Usage:$(RESET) make [target] [VAR=value]\n\n"
	@printf "  $(GREEN)make$(RESET)             run the full pipeline w/ system Python (split -> train -> predict)\n"
	@printf "  $(GREEN)make $(YELLOW)USE_VENV=1$(RESET)  run the same pipeline, but inside a .venv\n"
	@printf "  $(YELLOW)split$(RESET)            data.csv -> data_train.csv, data_valid.csv\n"
	@printf "  $(YELLOW)train$(RESET)            train the model, save .npz + learning curves\n"
	@printf "  $(YELLOW)predict$(RESET)          evaluate on the valid set, save predictions.csv\n"
	@printf "  $(YELLOW)deps$(RESET)             install requirements (into .venv when USE_VENV=1)\n"
	@printf "  $(YELLOW)clean$(RESET)            remove generated artifacts\n"
	@printf "  $(YELLOW)fclean$(RESET)           remove artifacts + .venv\n"
	@printf "  $(YELLOW)re$(RESET)               clean, then build everything\n"
	@printf "\n$(GRAY)Arguments$(RESET) (forwarded to each script; omit them to use the script\047s defaults):\n"
	@printf "  $(YELLOW)make split SPLIT_ARGS=\"...\"$(RESET)  -> srcs/split.py\n"
	@printf "      --valid-ratio R    rows held out for validation, as a fraction (e.g. 0.3)\n"
	@printf "      --seed N           RNG seed for the train/valid shuffle\n"
	@printf "  $(YELLOW)make train TRAIN_ARGS=\"...\"$(RESET)  -> srcs/train.py\n"
	@printf "      --layer N [N ...]  hidden-layer sizes (one value per hidden layer)\n"
	@printf "      --epochs N         number of training epochs\n"
	@printf "      --batch_size N     mini-batch size\n"
	@printf "      --learning_rate F  learning rate\n"
	@printf "  USAGE: make train TRAIN_ARGS=\"--layer 64 32 --epochs 60 --learning_rate 0.05 --batch_size 32\"\n\n"
	@printf "  $(YELLOW)make predict PREDICT_ARGS=\"...\"$(RESET)  -> srcs/predict.py\n"
	@printf "      --input PATH       CSV file to evaluate (default: the validation split)\n"
	@printf "\n  $(GRAY)Example:$(RESET) make train TRAIN_ARGS=\"--layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314\"\n"
	@printf "  $(GRAY)Full per-script flags:$(RESET) python3 srcs/train.py --help\n"
	@printf "\n"

# Create venv on demand. Tied to the literal .venv path, so this rule only ever
# builds .venv (never a stray rule for "python3"). Used only when USE_VENV=1.
$(VENV)/bin/python:
	@printf "$(GRAY)[venv]$(RESET) creating virtualenv at $(VENV)...\n"
	$(SYS_PY) -m venv $(VENV)
	$(VENV)/bin/pip install -U pip

deps: | $(VENV_TARGET)
	@printf "$(GRAY)[deps]$(RESET) installing requirements...\n"
	$(PIP) install -r requirements.txt

dirs:
	@mkdir -p $(DATASET_DIR) $(MODEL_DIR) $(OUTPUT_DIR)

# Optional CLI args passed through to each script (empty -> script defaults).
# e.g. make train TRAIN_ARGS="--layer 24 24 24 --epochs 84 --batch_size 8 --learning_rate 0.0314"
#      make split SPLIT_ARGS="--valid-ratio 0.3 --seed 7"
#      make predict PREDICT_ARGS="--input datasets/data.csv"
SPLIT_ARGS   ?=
TRAIN_ARGS   ?=
PREDICT_ARGS ?=

split: $(SPLIT_PY) | dirs $(PREP)
	@printf "$(PINK)[split]$(RESET) running...\n"
	$(PY) $(SPLIT_PY) $(SPLIT_ARGS)

train: $(TRAIN_PY) | dirs $(PREP)
	@printf "$(BLUE)[train]$(RESET) running...\n"
	$(PY) $(TRAIN_PY) $(TRAIN_ARGS)

predict: $(PREDICT_PY) | dirs $(PREP)
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
	@printf "$(GREEN)[fclean]$(RESET) removing artifacts and venv...\n"
	@rm -rf $(VENV)
	@rm -f \
		$(MODEL_DIR)/*.npz \
		$(MODEL_DIR)/latest \
		$(MODEL_DIR)/predictions.csv \
		$(OUTPUT_DIR)/*.png \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

re: clean all

.PHONY: all help deps dirs split train predict clean fclean re