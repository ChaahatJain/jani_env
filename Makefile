# -------- Config --------
IMAGE := garricklin/jani_env
SHA := $(shell git rev-parse --short HEAD)
BUILD_TIME := $(shell date -u +"%Y-%m-%dT%H:%M:%SZ")

# Optional: detect dirty repo
DIRTY := $(shell git diff --quiet || echo "-dirty")

FULL_SHA := $(SHA)$(DIRTY)

# Default target platform for your university cluster
PLATFORM ?= linux/amd64

# Detect whether buildx is available
HAVE_BUILDX := $(shell docker buildx version >/dev/null 2>&1 && echo yes || echo no)

# -------- Targets --------

.PHONY: print-version
print-version:
	@echo "Image:      $(IMAGE)"
	@echo "Git SHA:    $(FULL_SHA)"
	@echo "Build time: $(BUILD_TIME)"
	@echo "Platform:   $(PLATFORM)"
	@echo "Buildx:     $(HAVE_BUILDX)"


# Ensure a buildx builder exists and is selected (safe to run repeatedly)
.PHONY: buildx-init
buildx-init:
ifeq ($(HAVE_BUILDX),yes)
	@docker buildx inspect jani_builder >/dev/null 2>&1 || docker buildx create --name jani_builder --use >/dev/null
	@docker buildx use jani_builder >/dev/null
else
	@echo "ERROR: docker buildx not available. Update Docker or enable buildx." 1>&2
	@exit 1
endif


.PHONY: build
build:
ifeq ($(HAVE_BUILDX),yes)
	@$(MAKE) buildx-init
	docker buildx build --platform $(PLATFORM) --pull \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		--load \
		.
else
# Fallback: classic docker build (works on WSL2; on Apple Silicon this may build arm64)
	docker build --pull \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		.
endif


# Build and push to Docker Hub (recommended for cluster use)
.PHONY: push
push:
ifeq ($(HAVE_BUILDX),yes)
	@$(MAKE) buildx-init
	docker buildx build --platform $(PLATFORM) --pull \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		--push \
		.
else
	@echo "WARNING: buildx not available. Pushing whatever arch was built locally." 1>&2
	docker build --pull \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		.
	docker push $(IMAGE):latest
	docker push $(IMAGE):$(FULL_SHA)
endif


# No-cache variants
.PHONY: no-cache
no-cache:
ifeq ($(HAVE_BUILDX),yes)
	@$(MAKE) buildx-init
	docker buildx build --platform $(PLATFORM) --no-cache \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		--load \
		.
else
	docker build --no-cache \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		.
endif


.PHONY: no-cache-push
no-cache-push:
ifeq ($(HAVE_BUILDX),yes)
	@$(MAKE) buildx-init
	docker buildx build --platform $(PLATFORM) --no-cache \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		--push \
		.
else
	@echo "WARNING: buildx not available. Pushing whatever arch was built locally." 1>&2
	docker build --no-cache \
		--build-arg GIT_SHA=$(FULL_SHA) \
		--build-arg BUILD_TIME=$(BUILD_TIME) \
		-t $(IMAGE):latest \
		-t $(IMAGE):$(FULL_SHA) \
		.
	docker push $(IMAGE):latest
	docker push $(IMAGE):$(FULL_SHA)
endif


# Inspect what platforms Docker Hub advertises (handy sanity check)
.PHONY: inspect-remote
inspect-remote:
ifeq ($(HAVE_BUILDX),yes)
	docker buildx imagetools inspect $(IMAGE):latest
else
	@echo "ERROR: buildx not available (needed for imagetools inspect)." 1>&2
	@exit 1
endif