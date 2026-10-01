#!/bin/bash
CURDIR="$(pwd)"
if [ -d /models ]; then
	if [ ! -d /models/Qwen3.8-27B-Quark-AWQ-MXFP4 ]; then
	    (
		cd /models || exit
		git clone https://huggingface.co/amd/Qwen3.8-27B-Quark-AWQ-MXFP4
	    )
	fi
	if [ ! -d /models/Qwen3.8-27B-DFlash2-FP8 ]; then
	    (
		cd /models || exit
		git clone https://huggingface.co/tcclaviger/Qwen3.8-27B-DFlash2-FP8
	    )
	fi
	if [ ! -d /models/Qwen3.8-27B-MXFP4-mtpfp8 ]; then
	    mtp="$CURDIR/fp8_mtp.py"
	    (
		cd /models || exit
		python3 -m venv venv
		./venv/bin/pip install numpy torch --index-url https://download.pytorch.org/whl/cpu
		./venv/bin/python3 "$mtp" "Qwen3.8-27B-Quark-AWQ-MXFP4" "Qwen3.8-27B-MXFP4-mtpfp8"
		rm -rf venv
	    )
	fi
	if [ ! -d /models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken ]; then
		(
		cd /models || exit
		mkdir -p Qwen3.8-27B-MXFP4-mtpfp8-pertoken
		for f in model.safetensors tokenizer.json tokenizer_config.json merges.txt \
		    generation_config.json chat_template.jinja preprocessor_config.json processor_config.json; do
		    [ -e "/models/Qwen3.8-27B-MXFP4-mtpfp8/$f" ] && ln -sf "/models/Qwen3.8-27B-MXFP4-mtpfp8/$f" "/models/Qwen3.8-27B-MXFP4-mtpfp8-pertoken/$f"
		done
		python3 "$CURDIR/mtpfp8-pertoken.py" "Qwen3.8-27B-MXFP4-mtpfp8/config.json" "Qwen3.8-27B-MXFP4-mtpfp8-pertoken/config.json"
		)
	fi
else
	echo "/models does not exists..."
fi
if [ ! -e chat_template.jinja ]; then
	curl -fsSL https://huggingface.co/froggeric/Qwen-Fixed-Chat-Templates/resolve/main/chat_template.jinja -o chat_template.jinja
fi
