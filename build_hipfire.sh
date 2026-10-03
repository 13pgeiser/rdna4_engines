#!/bin/bash
(
	cd hipfire
	docker build -f Containerfile --target runtime -t local/hipfire .
)
