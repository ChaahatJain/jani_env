run:
	docker	run	-it	-v ~/gurobi_wls.lic:/opt/gurobi/gurobi.lic	-v ${PWD}:/jani_env	jani_in_python