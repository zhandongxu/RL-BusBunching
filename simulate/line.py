from .bus import Bus

class Generator:
    def __init__(self, line_info) -> None:
        self.__line_info = line_info

    def generate_buses(self):
        bus_init_loc_dict = self.__line_info['bus_init_loc']
        init_link_bus = {}
        for bus_id, init_loc in bus_init_loc_dict.items():
            init_link_bus[init_loc] = Bus(bus_id, self.__line_info)

        return init_link_bus