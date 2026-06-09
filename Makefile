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
PY     := $(VENV)/bin/python
PIP    := $(VENV)/bin/pip

# Project dirs
SRC_DIR     ?= srcs
DATASET_DIR ?= datasets
MODEL_DIR   ?= models
OUTPUT_DIR  ?= outputs

# Scripts
SPLIT_PY    := $(SRC_DIR)/split.py
TRAIN_PY    := $(SRC_DIR)/train.py
PREDICT_PY  := $(SRC_DIR)/predict.py

# Ensure imports work (e.g., from mlp_numpy import ...)
export PYTHONPATH := $(SRC_DIR):$(PYTHONPATH)

# Default pipeline
all: deps dirs split train predict


help:
	@printf "\nUsage:\n"
	@printf "	$(GREEN)make\n"                        # split -> train -> predict
	@printf "	$(GREEN)make $(YELLOW)split\n"                  # data.csv -> data_train.csv, data_valid.csv
	@printf "	$(GREEN)make $(YELLOW)train\n"                  # train model, save npz + learning curves
	@printf "	$(GREEN)make $(YELLOW)predict\n"                # evaluate on valid set, save predictions.csv
	@printf "	$(GREEN)make $(YELLOW)clean$(RESET)\n"          # remove artifacts
	@printf "	$(GREEN)make $(YELLOW)fclean$(RESET)\n"         # remove artifacts + venv
	@printf "\n"

# Create venv on demand. Other targets depend on this as an order-only prereq.
$(PY):
	@printf "$(GRAY)[venv]$(RESET) creating virtualenv at $(VENV)...\n"
	$(SYS_PY) -m venv $(VENV)
	$(PIP) install -U pip

deps: | $(PY)
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

split: $(SPLIT_PY) | dirs $(PY)
	@printf "$(PINK)[split]$(RESET) running...\n"
	$(PY) $(SPLIT_PY) $(SPLIT_ARGS)

train: $(TRAIN_PY) | dirs $(PY)
	@printf "$(BLUE)[train]$(RESET) running...\n"
	$(PY) $(TRAIN_PY) $(TRAIN_ARGS)

predict: $(PREDICT_PY) | dirs $(PY)
	@printf "$(YELLOW)[predict]$(RESET) running...\n"
	$(PY) $(PREDICT_PY) $(PREDICT_ARGS)

clean:
	@printf "$(GREEN)[clean]$(RESET) removing artifacts...\n"
	@rm -f \
		$(MODEL_DIR)/saved_model.npz \
		$(MODEL_DIR)/predictions.csv \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

fclean:
	@printf "$(GREEN)[fclean]$(RESET) removing artifacts and venv...\n"
	@rm -rf $(VENV)
	@rm -f \
		$(MODEL_DIR)/*.npz \
		$(MODEL_DIR)/predictions.csv \
		$(OUTPUT_DIR)/*.png \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

re: clean all

.PHONY: all help deps dirs split train predict clean fclean re
