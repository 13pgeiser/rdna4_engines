#!/bin/bash
TAG="${1:-qwen3.8:27b-mq4-xts}"
CURDIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# -------------------------------
# docker image (build only if missing)
# -------------------------------
IMAGE="local/hipfire"
if ! docker image inspect "$IMAGE" >/dev/null 2>&1; then
    (
	cd "$CURDIR/hipfire" || exit
	docker build -f Containerfile --target runtime -t "$IMAGE" .
    )
else
    echo "image $IMAGE already present, skipping build"
fi

# -------------------------------
# model (pulled into host /models)
# -------------------------------
if ! docker run --rm -v /models:/root/.hipfire/models "$IMAGE" list 2>/dev/null | grep -q "$TAG"; then
    docker run --rm -v /models:/root/.hipfire/models "$IMAGE" pull "$TAG"
fi

# -------------------------------
# run
# -------------------------------
sudo bash -c "echo "210000000" > $(echo /sys/class/drm/card0/device/hwmon/hwmon*/)power1_cap"
docker rm -f hipfire_server
docker run --name hipfire_server --network host --device /dev/kfd --device /dev/dri --security-opt seccomp=unconfined \
 -v /models:/root/.hipfire/models -v hipfire-kcache:/var/cache/hipfire "$IMAGE" serve "$TAG" 0.0.0.0:11435
