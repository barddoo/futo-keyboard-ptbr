PYTHON := uv run $(PYTHON)

.PHONY: all setup validate clean

# Language to build: a config in langs/<code>.py (pt, fr). Every stage reads
# it from the environment.  Usage: make KEYBOARD_LANG=fr
# (Not named LANG: that is the POSIX locale variable.)
KEYBOARD_LANG ?= pt
export KEYBOARD_LANG

# Paths come from the language config (stdlib-only, no venv needed)
langcfg = $(shell KEYBOARD_LANG=$(KEYBOARD_LANG) python3 -m langs $(1))
RAW_DIR       := $(call langcfg,raw_dir)
PROCESSED_DIR := $(call langcfg,processed_dir)
ifeq ($(RAW_DIR),)
$(error unknown KEYBOARD_LANG "$(KEYBOARD_LANG)" - see langs/)
endif

DATA_RAW   := $(RAW_DIR)/mc4_pt.txt
DATA_CLEAN := $(PROCESSED_DIR)/train.txt
CORRECTION := $(PROCESSED_DIR)/correction_train.txt
TOKENIZER  := $(call langcfg,sp_model)
CHECKPOINT := $(call langcfg,checkpoint_dir)/best.pt
F32_GGUF   := $(call langcfg,f32_gguf)
Q6K_GGUF   := $(call langcfg,q6k_gguf)
FINAL      := $(Q6K_GGUF)

all: $(FINAL) validate

setup:
	bash scripts/setup.sh

train-remote:
	AMD_HOST=$(AMD_HOST) bash scripts/sync_and_train.sh

$(DATA_RAW):
	$(PYTHON) 01_data/download.py

$(DATA_CLEAN): $(DATA_RAW)
	$(PYTHON) 01_data/prepare.py

# FUTO correction-format samples (autocorrect / swipe) generated from the corpus
$(CORRECTION): $(DATA_CLEAN)
	$(PYTHON) 01_data/make_correction_data.py

$(TOKENIZER): $(DATA_CLEAN)
	$(PYTHON) 01_data/train_tokenizer.py

$(CHECKPOINT): $(TOKENIZER) $(CORRECTION)
	$(PYTHON) 02_train/train.py

$(F32_GGUF): $(CHECKPOINT)
	$(PYTHON) 03_convert/export_to_gguf.py

$(FINAL): $(F32_GGUF)
	bash 03_convert/quantize.sh

# Validate the final GGUF against FUTO's loader requirements
validate: $(FINAL)
	$(PYTHON) 04_inject/inject_metadata.py

clean:
	rm -rf 01_data/raw 01_data/processed 01_data/tokenizer \
	       02_train/checkpoints \
	       03_convert/*.gguf
