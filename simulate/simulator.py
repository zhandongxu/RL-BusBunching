from .line import Generator
from .stop import Stop
from .link import Link
from .snapshot import Snapshot


class Simulator:
    '''Simulating bus operations along a loop corridor with a given control agent'''
    def __init__(self, config, agent=None, is_record_link_traje=False) -> None:
        # indicate if the link trajectory of each bus should be recorded to csv
        self.__is_record_link_traje = is_record_link_traje
        # RL control agent
        self.__agent = agent
        self.__config = config

        # simulation
        self.__dt = config.dt
        self.__sim_duration = config.sim_duration
        self.__gener = Generator(config.line_info)

        # init bus stops
        self.__stops = {}
        for stop_id, info in config.stop_info.items():
            stop = Stop(stop_id, info)
            self.__stops[stop_id] = stop

        # init links
        self.__links = {}
        for link_id, info in config.link_info.items():
            link = Link(link_id, info)
            self.__links[link_id] = link

        self.reset(episode=0)

    def reset(self, episode):
        self.__episode = episode
        self.__ct = 0.0
        for _, link in self.__links.items():
            link.reset()
        for _, stop in self.__stops.items():
            stop.reset()

        self.__buses = []
        init_link_bus = self.__gener.generate_buses()
        for init_link, bus in init_link_bus.items():
            self.__buses.append(bus)
            self.__links[init_link].enter_bus(self.__ct, bus, self.__episode)

    def get_buses_for_plot(self):
        return self.__buses

    def simulate(self, agent_config):
        while True:
            self.step()
            if self.__ct > self.__sim_duration:
                self.record()
                if self.__is_record_link_traje:
                    self.get_bus_link_trajectory(agent_config)
                break

    def step(self):
        # link operations
        for link_id, link in self.__links.items():
            leave_buses = link.forward(self.__ct, self.__dt)
            self.__transfer(leave_buses, link_id=link_id, stop_id=None)

        # stop operations
        for stop_id, stop in self.__stops.items():
            leave_buses = stop.operation(self.__ct, self.__dt)
            self.__transfer(leave_buses, link_id=None, stop_id=stop_id)

        # accumulate pax riding time for each time step
        for bus in self.__buses:
            bus.accumulate_pax_ride_time(self.__dt)

        self.__ct += self.__dt

    def __transfer(self, leave_buses, link_id=None, stop_id=None):
        for bus in leave_buses:
            if link_id is not None:
                next_stop_id = bus.get_next_stop(link_id)
                self.__stops[next_stop_id].enter_bus(self.__ct, bus)
                # if one trip (loop) is finished, count trip_no += 1
                if next_stop_id == 0:
                    bus.count_trip_no()
            # do action when entering the next link
            if stop_id is not None:
                next_link_id = bus.get_next_link(stop_id)
                self.__links[next_link_id].enter_bus(
                    self.__ct, bus, self.__episode)

                # take a snapshot and determine the holding time
                snapshot = self.__take_snapshot(bus.bus_id, stop_id)
                hold_time = self.__agent.cal_hold_time(snapshot)
                bus.hold_time = hold_time
                self.__agent.track_headway_deviations(snapshot)

    def __take_snapshot(self, curr_bus_id, curr_stop_id):
        '''Take a snapshot of the whole system when a bus departs from the stop'''
        snapshot = Snapshot(self.__episode, self.__ct, curr_bus_id,
                            curr_stop_id, self.__buses, self.__stops, self.__config)
        return snapshot

    def get_bus_link_trajectory(self, agent_config):
        for bus in self.__buses:
            bus.get_link_trajectory(agent_config)

    def record(self):
        # when simulation ends, record the wait time and arrival of each stop
        self.__agent.record_wait_time(self.get_wait_time_each_stop())
        self.__agent.record_arrival(self.get_arrival_each_stop())
        self.__agent.record_ride_time(self.get_ride_time_each_bus())
        self.__agent.record_boarding(self.get_boarding_each_bus())
        self.__agent.record_occupancy(self.get_occupy_each_stop())
        self.__agent.record_headway(self.get_headway_each_stop())

    def get_boarding_each_bus(self):
        boarding_dict = {}
        for bus in self.__buses:
            boarding_dict[bus.bus_id] = bus.cum_pax_board
        return boarding_dict

    def get_ride_time_each_bus(self):
        ride_time_dict = {}
        for bus in self.__buses:
            ride_time_dict[bus.bus_id] = bus.cum_pax_ride_time
        return ride_time_dict

    def get_arrival_each_stop(self):
        stop_arrival_dict = {}
        for stop_id, stop in self.__stops.items():
            stop_arrival_dict[stop_id] = stop.pax_total_arriv
        return stop_arrival_dict

    def get_wait_time_each_stop(self):
        stop_wait_time_dict = {}
        for stop_id, stop in self.__stops.items():
            stop_wait_time_dict[stop_id] = stop.pax_total_wait_time
        return stop_wait_time_dict

    def get_occupy_each_stop(self):
        stop_bus_load_dict = {}
        for stop_id, stop in self.__stops.items():
            stop_bus_load_dict[stop_id] = stop.bus_load
        return stop_bus_load_dict

    def get_headway_each_stop(self):
        stop_bus_headway_dict = {}
        for stop_id, stop in self.__stops.items():
            stop_bus_headway_dict[stop_id] = stop.stop_headway
        return stop_bus_headway_dict