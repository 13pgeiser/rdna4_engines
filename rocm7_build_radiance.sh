#!/bin/bash
source base_build_radiance.sh
(
	cd radiance-vllm-mxfp4
	docker build  -t "radiance-vllm-mxfp4:$(cat VERSION)" -f Dockerfile .
)

