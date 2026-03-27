import json
import random
import os
import shutil


class PythonUtils:

    # files

    @staticmethod
    def exists(path: str, as_file: bool = None) -> bool:
        if as_file is None:
            return os.path.exists(path)
        else:
            return os.path.isfile(path) if as_file else os.path.isdir(path)

    @staticmethod
    def extract_dir(path_to_file: str) -> str:
        return os.path.dirname(path_to_file)

    @staticmethod
    def extract_subdir(path_to_file: str) -> str:
        return os.path.split(PythonUtils.extract_dir(path_to_file))[1]

    @staticmethod
    def extract_basename(path_to_file: str) -> str:
        return os.path.basename(path_to_file)

    @staticmethod
    def split_basename(path_to_file: str) -> tuple:
        return os.path.splitext(PythonUtils.extract_basename(path_to_file))

    @staticmethod
    def extract_filename(path_to_file: str) -> str:
        return PythonUtils.split_basename(path_to_file)[0]

    @staticmethod
    def extract_ext(path_to_file: str) -> str:
        return PythonUtils.split_basename(path_to_file)[1]

    @staticmethod
    def read_file(path_to_file: str) -> str:
        with open(path_to_file, 'r') as f:
            return f.read()

    @staticmethod
    def read_lines(path_to_file: str) -> list:
        return [line for line in open(path_to_file, 'r')]

    @staticmethod
    def write_file(path_to_file: str, file_str: str, append: bool = False) -> None:
        PythonUtils.mkdir_if(PythonUtils.extract_dir(path_to_file))
        #
        with open(path_to_file, 'a' if append else 'w') as f:
            f.write(file_str)

    @staticmethod
    def copy_file(path_to_file: str, path_to_copy: str) -> None:
        shutil.copy(path_to_file, path_to_copy)

    @staticmethod
    def move_file(path_to_file: str, path_to_dest: str) -> None:
        shutil.move(path_to_file, path_to_dest)

    @staticmethod
    def mkdir_if(path_to_dir: str) -> None:
        if not path_to_dir == '' and not os.path.exists(path_to_dir):
            os.makedirs(path_to_dir)

    @staticmethod
    def mkdir_for_file(path_to_file: str) -> None:
        PythonUtils.mkdir_if(PythonUtils.extract_dir(path_to_file))

    @staticmethod
    def is_used_path(path_str: str, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> bool:
        required_strs = required_strs if required_strs is None or isinstance(required_strs, list) else [required_strs]
        ignore_strs = list() if ignore_strs is None else ignore_strs if isinstance(ignore_strs, list) else [ignore_strs]
        # required strings may be a list of list -> essentially "DNF"
        if required_strs is not None and not any([(all(req_sub_str in path_str for req_sub_str in required_str) if isinstance(required_str, list) else required_str in path_str) for required_str in required_strs]):
            return False
        if any([ignore_str in path_str for ignore_str in ignore_strs]):
            return False
        return not ignore_invalid_symlink or not os.path.islink(path_str) or PythonUtils.exists(path_str)

    @staticmethod
    def extract_directory_names(path_to_dir: str, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> list:
        return [sub_dir for sub_dir in os.listdir(path_to_dir) if os.path.isdir(os.path.join(path_to_dir, sub_dir)) and PythonUtils.is_used_path(os.path.join(path_to_dir, sub_dir), required_strs=required_strs, ignore_strs=ignore_strs, ignore_invalid_symlink=ignore_invalid_symlink)]

    @staticmethod
    def extract_directories(path_to_dir: str, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> list:
        return [os.path.join(path_to_dir, sub_dir) for sub_dir in PythonUtils.extract_directory_names(path_to_dir, required_strs=required_strs, ignore_strs=ignore_strs, ignore_invalid_symlink=ignore_invalid_symlink)]

    @staticmethod
    def extract_filenames(path_to_dir: str, file_ext: str = None, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> list:
        return [f for f in os.listdir(path_to_dir) if os.path.isfile(os.path.join(path_to_dir, f)) and (file_ext is None or f.endswith(file_ext)) and PythonUtils.is_used_path(os.path.join(path_to_dir, f), required_strs=required_strs, ignore_strs=ignore_strs, ignore_invalid_symlink=ignore_invalid_symlink)]

    @staticmethod
    def extract_files(path_to_dir: str, file_ext: str = None, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> list:
        return [os.path.join(path_to_dir, f) for f in PythonUtils.extract_filenames(path_to_dir, file_ext, required_strs=required_strs, ignore_strs=ignore_strs, ignore_invalid_symlink=ignore_invalid_symlink)]

    @staticmethod
    def extract_files_recursively(path_to_dir: str, file_ext: str = None, ignore_dir: list = None, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> list:
        ignore_dir = ignore_dir if isinstance(ignore_dir, list) else [ignore_dir]
        list_of_files = list()
        for path, current_subdir, files in os.walk(path_to_dir, topdown=True):
            current_subdir[:] = [subdir for subdir in current_subdir if ignore_dir is None or subdir not in ignore_dir]
            list_of_files += [os.path.join(path, f) for f in files if (file_ext is None or f.endswith(file_ext)) and PythonUtils.is_used_path(os.path.join(path, f), required_strs=required_strs, ignore_strs=ignore_strs, ignore_invalid_symlink=ignore_invalid_symlink)]
        return list_of_files

    @staticmethod
    def filter(list_of_files: list, file_ext: str = None, required_strs: list = None, ignore_strs: list = None, ignore_invalid_symlink: bool = True) -> list:
        list_of_files_filter = list()
        for file in list_of_files:
            if file_ext is not None and file_ext != PythonUtils.extract_ext(file):
                continue
            if not PythonUtils.is_used_path(file, required_strs=required_strs, ignore_strs=ignore_strs, ignore_invalid_symlink=ignore_invalid_symlink):
                continue
            list_of_files_filter.append(file)
        return list_of_files_filter

    @staticmethod
    def join_path(path_prefix: str, *sub_paths: str) -> str:
        return os.path.join(path_prefix, *sub_paths)

    @staticmethod
    def make_file_path(dir_name: str, basename: str) -> str:
        return PythonUtils.join_path(dir_name, basename)

    # str #############################################################################################################

    @staticmethod
    def substitute(input_strs, patterns, substitutions):
        is_list = isinstance(input_strs, list)
        patterns = patterns if isinstance(patterns, list) else [patterns]
        substitutions = substitutions if isinstance(substitutions, list) else [substitutions]
        assert len(patterns) == len(substitutions)
        input_strs_sub = input_strs if is_list else [input_strs]
        for pattern, substitution in zip(patterns, substitutions):
            input_strs_sub = [input_str.replace(pattern, substitution) for input_str in input_strs_sub]
        return input_strs_sub if is_list else input_strs_sub[0]

    @staticmethod
    def substitutes(input_str: str, patterns: list, substitutions: list) -> str:
        assert len(patterns) == len(substitutions)
        rlt = str(input_str)
        for pattern, substitution in zip(patterns, substitutions):
            rlt = PythonUtils.substitute(rlt, pattern, substitution)
        return rlt

    @staticmethod
    def count_substrings(path_to_file: str, substring: str) -> int:
        file_str = PythonUtils.read_file(path_to_file)
        return file_str.count(substring)

    @staticmethod
    def to_variable_format(str_in: str) -> str:
        str_out = str_in.lower()
        return PythonUtils.substitute(str_out, " ", "_")

    # https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse [July 2023]
    @staticmethod
    def str2bool(v) -> bool:
        if isinstance(v, bool):
            return v
        if v.lower() in ('yes', 'true', 't', 'y', '1'):
            return True
        elif v.lower() in ('no', 'false', 'f', 'n', '0'):
            return False
        else:
            raise Exception("Boolean value expected.")

    # list #############################################################################################################

    @staticmethod
    def concat_lists(list_structure, recursive: bool = True) -> list:
        if not isinstance(list_structure, list):
            return [list_structure]
        rlt_list = list()
        for list_sub_structure in list_structure:
            if recursive or not isinstance(list_sub_structure, list):
                rlt_list += PythonUtils.concat_lists(list_sub_structure, recursive=recursive)
            else:
                rlt_list += list_sub_structure
        return rlt_list

    @staticmethod
    def list_remove(list_of_elements: list, elements_to_remove: list, inplace: bool = False) -> list:
        if inplace:
            for elem in elements_to_remove:
                if elem in list_of_elements:
                    list_of_elements.remove(elem)
            return list_of_elements
        else:
            return [elem for elem in list_of_elements if elem not in elements_to_remove]

    @staticmethod
    def unfold_list_of_lists(list_of_lists: list, max_list_len):
        # sanity
        assert max_list_len >= max([len(sub_list) for sub_list in list_of_lists])
        #
        unfolded_list = list()
        for i in range(0, max_list_len):
            for sub_list in list_of_lists:
                if sub_list is not None and i < len(sub_list):  # may be None
                    unfolded_list.append(sub_list[i])
        return unfolded_list

    @staticmethod
    def intersection_list(l1, l2):
        l2_set = set(l2)
        return [elem for elem in l1 if elem in l2_set]

    @staticmethod
    def list_to_str(strs: list, concat_str: str = "_") -> str:
        if isinstance(strs, str):
            return strs
        else:
            list_str = ""
            for list_str_ in strs:
                list_str += concat_str + list_str_
            return list_str

    # json #############################################################################################################

    @staticmethod
    def is_numeric(value: json):
        return isinstance(value, int) or isinstance(value, float)

    @staticmethod
    def is_bool(value: json):
        return isinstance(value, bool)

    @staticmethod
    def is_constant_value(value: json):
        return PythonUtils.is_numeric(value=value) or PythonUtils.is_bool(value=value)

    @staticmethod
    def str_is_int(value: str) -> bool:
        try:
            int(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def str_is_float(value: str) -> bool:
        assert isinstance(value, str)
        try:
            float(value)
            return True
        except ValueError:
            return False

    @staticmethod
    def to_scientific(x: float, precision: int) -> str:
        return ("{:." + str(precision) + "E}").format(x)

    @staticmethod
    def round(value: float, precision: int = 2):
        return int(round(value, precision)) if precision == 0 else round(value, precision)

    @staticmethod
    def round_to_str(value: float, precision: int = 2):
        return str(PythonUtils.round(value=value, precision=precision))

    @staticmethod
    def load_json(path_to_file: str) -> json:
        try:
            with open(path_to_file, 'r') as f:
                f_json = json.load(f)
            return f_json
        except json.decoder.JSONDecodeError:
            with open(path_to_file, 'r', encoding='utf-8-sig') as f:
                f_json = json.load(f)
            return f_json

    @staticmethod
    def dump_json(json_structure: json) -> str:
        return json.dumps(json_structure, ensure_ascii=False, sort_keys=True, indent=4)

    @staticmethod
    def write_json(json_structure: json, path_to_output_file: str) -> None:
        PythonUtils.mkdir_if(PythonUtils.extract_dir(path_to_output_file))
        #
        with open(path_to_output_file, 'w', encoding='utf-8') as f:
            f.write(PythonUtils.dump_json(json_structure))

    @staticmethod
    def json_array_size(path_to_file: str, key: str) -> int:
        try:
            f_json = PythonUtils.load_json(path_to_file)
        except json.decoder.JSONDecodeError:  # deprecated as exception already handled by load_json
            print(path_to_file)  # debugging
            f_str = PythonUtils.read_file(path_to_file)
            counter = 0
            i = 0
            while f_str[i] != '{':  # start of json file
                counter += 1
                i += 1
            f_str = f_str[counter:]  # to handle issue with UTF-8 BOM
            f_json = json.loads(f_str)
        #
        return len(f_json[key])

    #

    # misc #############################################################################################################

    @staticmethod
    def update_dict(current: dict, update: dict, inplace: bool = True) -> dict:
        current = current if inplace else dict(current)
        for key, val in update.items():
            current[key] = val
        return current

    @staticmethod
    def merge_dicts(*dicts: dict):
        if len(dicts) == 1 and isinstance(dicts[0], list):
            return PythonUtils.merge_dicts(dicts[0])
        else:
            rlt = dict()
            for d in dicts:
                PythonUtils.update_dict(rlt, d, inplace=True)
        return rlt

    @staticmethod
    def sum_dict(state: dict, keys):
        return sum([state[key] for key in keys if key in state])

    #

    @staticmethod
    def set_cond(cond_list: list, alt=None):
        for cond, value in cond_list:
            if cond:
                return value
        return alt

    @staticmethod
    def sum_up_cond(cond_list: list, base):
        for cond_val in cond_list:
            assert len(cond_val) == 2 or len(cond_val) == 3
            if len(cond_val) == 3:
                base += (cond_val[1] if cond_val[0] else cond_val[2])
            elif cond_val[0]:
                base += cond_val[1]
        return base

    #

    @staticmethod
    def gen_cond_str(str_cond: list, base: str = "", cond: bool = True):
        return PythonUtils.sum_up_cond(str_cond, base) if cond else ""

    @staticmethod
    def concat_to_str(str_list: list, sep: str = " "):
        rlt = ""
        for elem in str_list:
            rlt += sep + str(elem)
        return rlt

    #

    @staticmethod
    def range_inclusive(lower_bound: int, upper_bound: int):
        return range(lower_bound, upper_bound + 1)

    @staticmethod
    def generate_random_vector(size: int, total: int, lower: int = 0) -> list:
        assert total >= 0
        indexes = list(range(0, size))
        values = [0 for _ in range(0, size)]
        random.shuffle(indexes)
        for index in indexes:
            val = random.randint(lower, total)
            total -= val
            values[index] = val
            if total == 0:
                break
        if total > 0:  # randomly assign remainder
            index = random.choice(indexes)
            values[index] += total
        return values

    #

    @staticmethod
    def set_if_not_none(value, alt):
        return value if value is not None else alt

    @staticmethod
    def set_alt_if_none(value, alt):
        return alt if value is None else value

    @staticmethod
    def set_copy_if_not_none(value, alt_value):
        return value.copy() if value is not None else alt_value

    @staticmethod
    def set_alt_copy_if_none(value, alt_value):
        return alt_value.copy() if value is None else value
