IMAGE = jani_in_python
DOCKERHUB_USER = chaahatjain
build:
	docker build \
		--build-arg GIT_SHA=$(shell git rev-parse --short HEAD) \
		--build-arg BUILD_TIME=$(shell date -u +%Y-%m-%dT%H:%M:%SZ) \
		-t $(IMAGE) .

run:
	docker run -it \
		-e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic \
		-v ~/gurobi_wls.lic:/opt/gurobi/gurobi.lic \
		-v $(PWD):/jani_env \
		-v $(IMAGE)_engine_build:/jani_env/jani/engine/build \
		$(IMAGE)

push:
	docker tag $(IMAGE) $(DOCKERHUB_USER)/$(IMAGE):latest
	docker push $(DOCKERHUB_USER)/$(IMAGE):latest

logs_pull:
	ssh -q jain@conduit.hpc.uni-saarland.de "bash --norc --noprofile -c 'find /home/jain/jani_env/artifacts/pipeline -type d -name repair_logs | tar czf - -T -'" | tar xzf - -C cluster/pipeline/local_repair_logs/

