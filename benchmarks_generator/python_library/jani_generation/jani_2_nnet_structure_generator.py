class Jani2NNetStructureGenerator(object):

    @staticmethod
    def generate_automaton_instance(name: str, index: int):
        return {"name": name, "index": index}

    @staticmethod
    def generate_input_feature_to_value_variable(automaton_instance, name: str):
        return {"automaton": automaton_instance, "name": name}

    @staticmethod
    def generate_input_feature_to_global_value_variable(name: str):
        return Jani2NNetStructureGenerator.generate_input_feature_to_value_variable(None, name)

    @staticmethod
    def generate_edge(automaton_instance, index: int):
        return {"automaton": automaton_instance, "index": index}

    @staticmethod
    def generate_choice_structure(index, edges: list):
        return {"index": index, "edges": edges}

    @staticmethod
    def generate_jani2nnet_file(file: str, applicability_filtering: bool, input_features: list, elements: list, output_features: list):
        return {"file": file, "filter": applicability_filtering, "input": input_features, "elements": elements, "output": output_features}
