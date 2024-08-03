from .passenger import Passenger
from .bus import Bus
import numpy as np


class Stop:
    def __init__(self, stop_id, info) -> None:
        # config parameters
        self.__stop_id = stop_id
        self.__berth_num = info["berth_num"]
        self.__queue_rule = info["queue_rule"]
        self.__loc = info["loc"]
        self.__total_stop_num = info["total_stop_num"]

        dest_alight_prob = {}
        for count, p in enumerate(info['alight_probs']):
            dest_stop_id = (count + stop_id + 1) % self.__total_stop_num
            dest_alight_prob[dest_stop_id] = p

        self.__pax_queue = Passenger(info["pax_arriv_rate"], dest_alight_prob)

        self.reset()

    def reset(self):
        # operation variables
        self.__entry_queue = []
        self.__buses_in_berth = [None] * self.__berth_num

        # accumulate pax arrivals
        self.pax_total_arriv = 0.0
        # accumulate pax delays
        self.pax_total_wait_time = 0.0
        # clear pax queue
        self.__pax_queue.reset()
        self.pax_on_stop_num = self.__pax_queue.get_stop_pax_num()
        # recording variables
        self.last_depar_times = [0.0]
        # record bus load
        self.bus_load = []
        # record headway
        self.stop_headway = []

    def operation(self, ct, dt):
        arriv_pax = self.__pax_arriving(dt)

        self.__queueing()
        self.__dwelling(ct, dt)
        leave_buses = self.__leaving(ct)

        # accumulate pax arrivals
        self.pax_total_arriv += arriv_pax
        # update total pax wait time at each delta t
        self.pax_total_wait_time += self.__pax_queue.get_wait_time_per_delta(dt)
        # store all headway of every stop
        self.stop_headway = np.diff(self.last_depar_times)

        return leave_buses

    def enter_bus(self, ct, bus):
        bus.update_loc(ct, "stop", self.__stop_id, self.__loc)
        self.__entry_queue.append(bus)

    def __pax_arriving(self, dt):
        return self.__pax_queue.arrive(dt)

    def __queueing(self):
        if len(self.__entry_queue) == 0:
            return
        bus = self.__entry_queue[0]
        target_berth = self.__check_in()
        if target_berth >= 0:  # has available berth, enter
            self.__buses_in_berth[target_berth] = bus
            self.__entry_queue.pop(0)

    def __dwelling(self, ct, dt):
        for berth in range(len(self.__buses_in_berth) - 1, -1, -1):
            bus = self.__buses_in_berth[berth]
            if bus is None:
                continue
            bus.update_loc(ct, "stop", self.__stop_id, self.__loc)
            # alighting and boardings are done parallelly?
            # alighting and boardings are done sequentially
            # alight
            if bus.get_onboard_amount(self.__stop_id) > 0:
                bus.alight(self.__stop_id, dt)

            # if bus.get_onboard_amount(self.__stop_id) <= 0:
            # board pax
            dest, amoun_board = self.__pax_queue.board(bus, dt)
            # dest and amoun_board are None if no pax can board (either queue is empty or bus is full)
            if dest is not None:
                bus.board(dest, amoun_board)
        # store total pax number on this stop after board
        self.pax_on_stop_num = self.__pax_queue.get_stop_pax_num()

    def __leaving(self, ct):
        leave_buses = []
        for berth in range(len(self.__buses_in_berth) - 1, -1, -1):
            bus = self.__buses_in_berth[berth]
            # if berth is empty, continue
            if bus is None:
                continue

            # if any onboard pax has not alighted, continue
            if bus.get_onboard_amount(self.__stop_id) > 0:
                continue

            # check if boarding ends
            # if capacity is enough and queue is not empty, the bus will not leave
            if bus.get_remaining_capacity() > 0.0 and not self.__pax_queue.check_pax_clear():
                continue

            # check queueing rule
            if self.__check_out(berth):
                leave_buses.append(bus)
                self.last_depar_times.append(ct)
                self.__buses_in_berth[berth] = None
                onboard_num = bus.get_onboard_num()
                self.bus_load.append(onboard_num)

        return leave_buses

    def __check_in(self):
        if self.__queue_rule == "FCFG":
            # find the most downstream non-empty stop, and directly move to it
            for b in range(len(self.__buses_in_berth)):
                if self.__buses_in_berth[b] == None:
                    target_berth = b
                    return target_berth
            else:
                print(
                    "Need more berth_num to ensure that there is awalys enough berth!"
                )
        else:  # not the unlimited case
            target_berth = -1  # negative means no berth is available
            for b in range(len(self.__buses_in_berth) - 1, -1, -1):
                if self.__buses_in_berth[b] == None:
                    target_berth = b
                else:
                    break
            return target_berth

    def __check_out(self, which_berth):
        if self.__queue_rule == "FCFG":
            return True
        else:  # not the unlimited case
            if which_berth == 0:  # most downstream berth, directly leave
                return True
            for b in range(which_berth - 1, -1, -1):
                if self.__buses_in_berth[b] != None:
                    break
                else:
                    if b == 0:  # all the most downstream berths are clear
                        return True
            return False