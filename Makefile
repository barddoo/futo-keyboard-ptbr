PYTHON := uv run $(PYTHON)

.PHONY: all setup validate clean

DATA_RAW   := 01_data/raw/mc4_pt.txt
DATA_CLEAN := 01_data/processed/train.txt
CORRECTION := 01_data/processed/correction_train.txt
TOKENIZER  := 01_data/tokenizer/pt_br_keyboard.model
CHECKPOINT := 02_train/checkpoints/best.pt
F32_GGUF   := 03_convert/pt_br_keyboard_f32.gguf
Q6K_GGUF   := 03_convert/pt_br_keyboard_q6k.gguf
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
