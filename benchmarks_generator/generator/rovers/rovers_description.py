import json
import re

from python_utils import PythonUtils


class RoversDescription(object):
    DIRECTION_NAMES = ["down", "up", "left", "right"]
    DIRECTION_DIFFS = dict([("down", (0, -1)), ("up", (0, 1)), ("left", (-1, 0)), ("right", (1, 0))])  # (dx, dy)

    class Cell:
        def __init__(self, x: int, y: int):
            self.x = x
            self.y = y

        def to_str(self) -> str:
            return "x" + str(self.x) + "_y" + str(self.y)

        def __eq__(self, other):
            return isinstance(other, RoversDescription.Cell) and self.x == other.x and self.y == other.y

        def __hash__(self):
            return 31 * self.y + self.x

    class Rover:
        INDEX = 0

        def __init__(self, battery: int, capacity: int):
            self.battery = battery
            self.capacity = capacity
            self.index = self.INDEX
            self.INDEX += 1

        def to_str(self) -> str:
            return "rover_" + str(self.index)

    class PathCapacity:
        def __init__(self, x: int, y: int, capacity: int):
            self.x = x
            self.y = y
            self.capacity = capacity

    def __init__(self, path_to_file: str):
        self.description = PythonUtils.load_json(path_to_file)
        self.name = self.description["name"]
        self.rovers = [self.Rover(rover["battery"], rover["capacity"]) for rover in self.description["rovers"]]
        self.load_grid()
        self.load_locations()

    ####################################################################################################################

    # parser aux:

    # noinspection PyAttributeOutsideInit
    def load_grid(self):
        json_grid = self.description["map"]
        assert len(json_grid) > 0
        self.x_dim = len(json_grid[0])
        self.y_dim = len(json_grid)
        self.lander = None
        self.charger = list()
        self.objectives = list()
        self.rocks = list()
        self.soil = list()
        # parse map
        y = 0
        for line in json_grid:
            assert len(line) == self.x_dim
            x = 0
            for cell in line:
                self.set_if_lander_position(cell, x, y)
                self.set_if_charger(cell, x, y)
                self.set_if_objective(cell, x, y)
                self.set_if_rock(cell, x, y)
                self.set_if_soil(cell, x, y)
                x += 1
            y += 1

        assert len(self.charger) == 1 and self.charger.__eq__(self.lander)  # assumed in some routines

    # noinspection PyAttributeOutsideInit
    def load_locations(self):
        self.default_path_capacity = None
        self.per_direction_capacities = dict()
        if "path-capacity" in self.description:
            path_capacity = self.description["path-capacity"]
            self.default_path_capacity = path_capacity["default"]
            self.per_direction_capacities = dict()
            if "per-direction" in path_capacity:
                per_direction = path_capacity["per-direction"]
                for direction in self.DIRECTION_NAMES:
                    if direction not in per_direction:
                        continue
                    self.per_direction_capacities[direction] = dict()
                    per_cap = self.per_direction_capacities[direction]
                    for cap_structure in per_direction[direction]:
                        if cap_structure["capacity"] == self.default_path_capacity:
                            continue
                        per_cap[cap_structure["capacity"]] = [self.Cell(src["x"], src["y"]) for src in cap_structure["srcs"]]

    # @staticmethod
    # def split_line(line: str):
    #    return line.split(" ")

    # @staticmethod
    # def cell_to_id(cell: tuple):
    #     return "x" + str(cell[0]) + "y" + str(cell[1])

    # def to_explicit(self, explicit_struct: json):
        #     explicit_struct["lander"] = self.cell_to_id(self.lander)
        #     explicit_struct["charger"] = [self.cell_to_id(cell) for cell in self.charger]
        #     explicit_struct["rocks"] = [{"loc": self.cell_to_id(cell), "amount": amount} for cell, amount in self.rocks]
        #     explicit_struct["soil"] = [{"loc": self.cell_to_id(cell), "amount": amount} for cell, amount in self.soil]

    # noinspection PyAttributeOutsideInit
    def set_if_lander_position(self, cell: json, x, y) -> None:
        if cell != 0 and "L" in cell:
            assert self.lander is None
            self.lander = self.Cell(x, y)

    def set_if_charger(self, cell: json, x, y) -> None:
        if cell != 0 and "C" in cell:
            self.charger.append(self.Cell(x, y))

    def set_if_objective(self, cell: json, x, y) -> None:
        if cell != 0 and "O" in cell:
            self.objectives.append(self.Cell(x, y))

    def set_if_rock(self, cell: json, x, y) -> None:
        return self.set_if(cell, x, y, "R", self.rocks)

    def set_if_soil(self, cell: json, x, y) -> None:
        return self.set_if(cell, x, y, "S", self.soil)

    @staticmethod
    def set_if(cell: json, x, y, type_str: str, type_list: list) -> None:
        if cell != 0 and type_str in cell:
            possible_ints = re.findall(r'\d+', cell.split(type_str)[1])
            amount = int(possible_ints[0] if len(possible_ints) > 0 else 1)
            type_list.append((RoversDescription.Cell(x, y), amount))

    ####################################################################################################################

    # interface:

    # map

    def num_cells(self):
        return self.x_dim * self.y_dim

    def range_cells(self) -> list:
        return [self.Cell(x, y) for x in range(0, self.x_dim) for y in range(0, self.y_dim)]

    def cell_to_id(self, cell: Cell) -> int:
        return self.cell_coordinates_to_id(cell.x, cell.y)

    def cell_coordinates_to_id(self, x, y) -> int:
        return x + y * self.x_dim

    # path capacity

    def extract_default_cap(self) -> int:
        return self.default_path_capacity

    def extract_cells(self, direction: str) -> dict:
        return self.per_direction_capacities[direction] if direction in self.per_direction_capacities else dict()

    def extract_min_cap(self, direction: str = None) -> int:
        if direction is None:
            return min([self.extract_min_cap(dir) for dir in self.DIRECTION_NAMES])
        else:
            return min([self.extract_default_cap()] + list(self.extract_cells(direction).keys()))

    def extract_special_cap_cells(self, direction: str = None) -> list:
        if direction is None:
            cells = list()
            for direction in self.DIRECTION_NAMES:
                cells += self.extract_special_cap_cells(direction)
            return cells
        return list() if direction not in self.per_direction_capacities.keys() else PythonUtils.concat_lists(list(self.per_direction_capacities[direction].values()))

    def extract_min_cap_for_cell(self, cell: Cell):
        min_caps = [self.default_path_capacity]
        for direction in self.DIRECTION_NAMES:
            for cap, cells in self.per_direction_capacities[direction].items():
                if cell in cells:
                    min_caps.append(cap)
        return min(min_caps)

    def has_default_cap(self) -> bool:
        return self.default_path_capacity is not None

    def has_special_cap(self) -> bool:
        return len(self.extract_special_cap_cells()) > 0

    def has_only_default_cap(self) -> bool:
        return self.has_default_cap() and not self.has_special_cap()

    # lander

    def lander_x(self) -> int:
        return self.lander.x

    def lander_y(self) -> int:
        return self.lander.y

    # rovers

    def num_rovers(self) -> int:
        return len(self.rovers)

    def range_rovers(self) -> list:
        return self.rovers

    # rocks & soil

    def rocks_on_cell(self, cell: tuple) -> int:
        amount = 0
        for rock in self.rocks:
            amount += rock[1] if cell == rock[0] else 0
        return amount

    def soil_on_cell(self, cell: tuple) -> int:
        amount = 0
        for soil in self.soil:
            amount += soil[1] if cell == soil[0] else 0
        return amount

    def rocks_in_total(self) -> int:
        amount = 0
        for rock in self.rocks:
            amount += rock[1]
        return amount

    def soil_in_total(self) -> int:
        amount = 0
        for soil in self.soil:
            amount += soil[1]
        return amount

    def samples_in_total(self) -> int:
        return self.rocks_in_total() + self.soil_in_total()

    def has_rocks(self) -> bool:
        return self.rocks_in_total() > 0

    def has_soil(self) -> bool:
        return self.soil_in_total() > 0

    def has_samples(self) -> bool:
        return self.has_rocks() or self.has_soil()

    # image

    def num_objectives(self) -> int:
        return len(self.objectives)

    def has_objectives(self) -> bool:
        return self.num_objectives() > 0

    # energy

    def get_energy(self) -> json:
        return self.description["energy-consumption"]

    def move_energy(self) -> int:
        return self.get_energy()["move"]

    def move_energy_per_rock(self) -> int:
        return self.move_energy()

    def move_energy_per_soil(self) -> int:
        return self.move_energy()

    def sample_energy(self) -> int:
        return self.get_energy()["sample"]

    def drop_energy(self) -> int:
        assert self.get_energy()["drop"] == 0  # so far expected and used in some routines
        return self.get_energy()["drop"]

    def take_image_energy(self) -> int:
        return self.get_energy()["take-image"]

    def share_image_energy(self) -> int:
        assert "share-image" not in self.get_energy()
        return self.get_energy()["share-image"] if "share-image" in self.get_energy() else 0

    def max_energy_per_step(self) -> int:
        # compute max move energy:
        max_rover_cap = max([rover.capacity for rover in self.rovers])
        if self.move_energy_per_rock() > self.move_energy_per_soil():
            num_rocks = max(max_rover_cap, self.rocks_in_total())
            num_soil = max(max_rover_cap - num_rocks, self.soil_in_total())
        else:
            num_soil = max(max_rover_cap, self.soil_in_total())
            num_rocks = max(max_rover_cap - num_soil, self.rocks_in_total())
        max_move_energy = self.move_energy() + num_rocks * self.move_energy_per_rock() + num_soil * self.move_energy_per_soil()
        #
        return max(max_move_energy, self.sample_energy(), self.drop_energy(), self.take_image_energy(), self.share_image_energy())


RoversDescription("description_files/mars.json")
