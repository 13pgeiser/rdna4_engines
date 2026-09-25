#!/bin/bash
(
	cd llama.cpp
	docker build -f .devops/rocm.Dockerfile -t local/llama.cpp --build-arg AMDGPU_TARGETS=gfx1201 .
)

