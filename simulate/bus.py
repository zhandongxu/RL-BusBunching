import csv


class Bus:
    def __init__(self, bus_id, line_info) -> None:
        self.bus_id = bus_id
        self.hold_time = None
        self.traje = {}

        self.__link_next_stop = line_info['link_next_stop']
        self.__stop_next_link = line_info['stop_next_link']
        self.__board_rate = line_info['board_rate']
        self.__alight_rate = line_info['alight_rate']
        self.__stop_num = line_info['stop_num']
        self.__capac = line_info['capacity']

        # record bus trajecries when entering a link
        self.__link_traje_record = []

        # bus offset from stop 0
        self.__relat_x = 0.0
        # speed for traversing the link
        self.__speed = None
        # travel time for traversing the link
        self.__trave_time = None
        # current spot type, 'link', or 'stop'
        self.__spot_type = None
        # current spot id
        self.__spot_id = -1
        # current trip number
        self.__trip_no = 0

        # store onboard pax for each destination
        self.__dest_pax = {dest: 0.0 for dest in range(self.__stop_num)}
        # store total onboard pax number
        self.onboard_num = sum(self.__dest_pax.values())

        # store cumulative pax boardings
        self.cum_pax_board = 0.0
        # store cumulative pax riding time
        self.cum_pax_ride_time = 0.0

    def get_link_trajectory(self, agent_config):
        with open('link_trajectory/' + agent_config["agent_name"] + '_link_trajectory.csv', 'a', encoding='UTF8', newline='') as f:
            writer = csv.writer(f)
            for row in self.__link_traje_record:
                writer.writerow(row)

    def record_link_trajectory(self, ep, bus_id, stop_id, ct, sampl_time):
        self.__link_traje_record.append([ep, bus_id, stop_id, ct, sampl_time])

    def get_next_link(self, stop_id):
        return self.__stop_next_link[stop_id]

    def get_next_stop(self, link_id):
        return self.__link_next_stop[link_id]

    def get_onboard_num(self):
        return sum(self.__dest_pax.values())

    def get_remaining_capacity(self):
        return self.__capac - sum(self.__dest_pax.values())

    def accumulate_pax_ride_time(self, dt):
        self.cum_pax_ride_time += sum(self.__dest_pax.values()) * dt

    def board(self, dest, amoun_board):
        self.__dest_pax[dest] += amoun_board
        self.cum_pax_board += amoun_board

    def get_onboard_amount(self, dest):
        return self.__dest_pax[dest]

    def alight(self, dest, dt):
        self.__dest_pax[dest] -= min(self.__dest_pax[dest], self.__alight_rate * dt)

    def update_loc(self, ct, spot_type, spot_id, relat_x):
        self.__spot_type = spot_type
        self.__spot_id = spot_id
        self.__relat_x = relat_x
        self.traje[ct] = {"trip_no": self.__trip_no, "spot_type": spot_type,
                          "spot_id": spot_id, "relat_x": relat_x}

    def get_bus_loc_info(self):
        return {'spot_type': self.__spot_type, 'spot_id': self.__spot_id, 'x': self.__relat_x}

    def get_next_loc_on_link(self, link_end_x, dt):
        x = min(link_end_x, self.__relat_x + self.__speed*dt)
        return x

    def set_speed(self, speed):
        self.__speed = speed

    def reduce_hold_time(self, dt):
        self.hold_time -= dt
        if self.hold_time <= 0.0:
            self.hold_time = None

    def count_trip_no(self):
        self.__trip_no += 1

    def get_trip_no(self):
        return self.__trip_no

    @property
    def board_rate(self):
        return self.__board_rate

    @property
    def relat_x(self):
        return self.__relat_x

    def set_travel_time(self, trave_time):
        self.__trave_time = trave_time

    def get_travel_time(self):
        return self.__trave_time
