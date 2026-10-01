#!/bin/bash
source base_build_radiance.sh
(
	cd radiance-vllm-mxfp4
	cat Dockerfile \
		| sed 's|^ARG ROCM_BASE=.*$|ARG ROCM_BASE=docker.io/rocm/dev-ubuntu-24.04:10.0.0-full|' \
		| sed 's|^ARG RELEASE_BASE=.*$|ARG RELEASE_BASE=docker.io/library/ubuntu:24.04|' \
		>Dockerfile.rocm10
	docker build  -t "rocm10-radiance-vllm-mxfp4:$(cat VERSION)" -f Dockerfile.rocm10 .
	rm -f Dockerfile.rocm10
)
