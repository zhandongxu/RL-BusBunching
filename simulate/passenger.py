import numpy as np
import random


class Passenger:
    def __init__(self, arriv_rate, dest_aligh_probs) -> None:
        self.__total_arriv_rate = arriv_rate
        # destination -> demand arrival rate
        self.__dest_rate = {dest: self.__total_arriv_rate *
                            p for dest, p in dest_aligh_probs.items()}
        self.reset()

    def reset(self):
        # destination -> queue length
        self.__dest_queue = {dest: 0.0 for dest, _ in self.__dest_rate.items()}

    def arrive(self, dt):
        total_arr_pax_this_delta = 0.0
        for dest, rate in self.__dest_rate.items():
            arr_pax = np.random.poisson(rate * dt / 60.0)
            self.__dest_queue[dest] += arr_pax
            total_arr_pax_this_delta += arr_pax
        return total_arr_pax_this_delta

    def get_wait_time_per_delta(self, dt):
        return sum(self.__dest_queue.values()) * dt

    def get_stop_pax_num(self):
        return sum(self.__dest_queue.values())

    def check_pax_clear(self):
        for dest, queue in self.__dest_queue.items():
            if queue > 0:
                return False
        else:
            return True

    def board(self, bus, dt):
        dests = list(self.__dest_queue.keys())
        random.shuffle(dests)
        for dest in dests:
            # if the queue has no pax, do nothing
            if self.__dest_queue[dest] == 0:
                continue
            # boarding
            remai_capac = bus.get_remaining_capacity()
            if remai_capac == 0.0:
                return None, None

            quota = min(bus.board_rate * dt, remai_capac)

            if self.__dest_queue[dest] >= quota:
                self.__dest_queue[dest] -= quota
                return dest, quota
            else:
                amoun_board = self.__dest_queue[dest]
                self.__dest_queue[dest] = 0
                return dest, amoun_board
        else:
            return None, None