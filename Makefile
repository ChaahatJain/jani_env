build:
	docker build -t jani_in_python .
run:
	docker	run	-it -e GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic -v ~/gurobi_wls.lic:/opt/gurobi/gurobi.lic	-v ${PWD}:/jani_env	jani_in_python