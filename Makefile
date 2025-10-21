SHELL := /bin/bash

RESET  := \033[0m
YELLOW := \033[33m
GREEN  := \033[32m
GRAY   := \033[90m
BLUE   := \033[94m

PY ?= python3

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
	@printf "	$(GREEN)make $(YELLOW)clean$(RESET)\n"                  # remove artifacts
	@printf "\n"

deps:
	$(PY) -m pip install -U pip
	$(PY) -m pip install -r requirements.txt

dirs:
	@mkdir -p $(DATASET_DIR) $(MODEL_DIR) $(OUTPUT_DIR)

split: $(SPLIT_PY) | dirs
	@printf "$(GREEN)[split]$(RESET) running...\n"
	$(PY) $(SPLIT_PY)

train: $(TRAIN_PY) | dirs
	@printf "$(GREEN)[train]$(RESET) running...\n"
	$(PY) $(TRAIN_PY)

predict: $(PREDICT_PY) | dirs
	@printf "$(GREEN)[predict]$(RESET) running...\n"
	$(PY) $(PREDICT_PY)

clean:
	@printf "$(GREEN)[clean]$(RESET) removing artifacts...\n"
	@rm -f \
		$(MODEL_DIR)/saved_model.npz \
		$(MODEL_DIR)/predictions.csv \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

fclean:
	@printf "$(GREEN)[clean]$(RESET) removing artifacts...\n"
	@rm -f \
		$(MODEL_DIR)/saved_model.npz \
		$(MODEL_DIR)/predictions.csv \
		$(OUTPUT_DIR)/loss.png \
		$(OUTPUT_DIR)/accuracy.png \
		$(DATASET_DIR)/data_train.csv \
		$(DATASET_DIR)/data_valid.csv

re: clean all

.PHONY: all help dirs split train predict clean re
