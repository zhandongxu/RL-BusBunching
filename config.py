from itertools import accumulate
from collections import defaultdict
from sympy import Symbol, nsolve
import pandas as pd

board_rate = 0.33   #pax/s
alight_rate = 0.55  #pax/s

class Config:
    def __init__(self, line):
        self.sim_duration = 2 * 3600   # simulation duration time (second)
        self.dt = 1.0                  # simulation step time (second)

        output_file = f"../data/{line}/" + str(line) + "output.csv"
        self.processing_data = pd.read_csv(output_file)
        self.stop_num = self.processing_data["seq"].max()
        self.processing_data["seq"] = self.processing_data["seq"] - 1
        self.bus_num = int(self.stop_num / 2.5) +1  #for 27/34
        self.bus_init_loc = [int(i * 2.5) for i in range(self.bus_num)]  #for 27/34
        self.bus_init_loc_dict = {bus_id: bus_loc for bus_id, bus_loc in enumerate(self.bus_init_loc)}
        self.link_length = self.processing_data["link_length"].tolist()
        self.link_num = len(self.link_length)

        self.mean_speed = 30/3.6

        self.get_stop_info(line)
        self.get_link_info()
        self.get_stop_beta()
        self.get_equ_headway()
        self.get_line_info()

    def get_stop_info(self, line):
        alight_dest_num = round(self.stop_num - 1 / 2)  # Since it is a loop route, it is assumed that the destination of the boarding passenger 
                                                        # is the next half stops from the stop of boarding.
        alight_probs = [0.0206, 0.0243, 0.0426, 0.0554, 0.0555, 0.0667, 0.0667, 0.0594, 0.0583, 0.0647, 0.0623,
                       0.0602, 0.0655, 0.0541, 0.0537, 0.0471, 0.0347, 0.0304, 0.0273, 0.0206, 0.0177]  # Randomising destination probabilities

        assert alight_dest_num > len(alight_probs), 'alight nums error'
        alight_probs.append(1 - sum(alight_probs))
        assert alight_probs[-1] > 0, 'alight probs error'
        self.alight_probs = tuple(alight_probs)

        if line in [34, 27, 'K2']:
            self.stop_pax_arriv_rate = self.processing_data.set_index('seq')['lambda'].to_dict()
        else:
            print("no information")

        self.stop_info = defaultdict(dict)
        milepost = list(accumulate(self.link_length))
        milepost.insert(0, 0)
        milepost.pop()
        for n in range(self.stop_num):
            self.stop_info[n]["loc"] = milepost[n]  # Milepost of station, the milepost of first station is 0
            self.stop_info[n]["pax_arriv_rate"] = self.stop_pax_arriv_rate[n] # Passenger arrival rate
            self.stop_info[n]["alight_probs"] = self.alight_probs # Probability distribution of passenger destinations
            self.stop_info[n]["berth_num"] = 6  # Number of platform berths
            self.stop_info[n]["total_stop_num"] = self.stop_num
            self.stop_info[n]["queue_rule"] = "FCFG"  # fist come first go

    def get_link_info(self):
        end_milepost = list(accumulate(self.link_length))
        start_milepost = [x - y for x, y in zip(end_milepost, self.link_length)]
        data = self.processing_data.to_dict(orient="records")
        link_info_dict = {d["seq"]: {"length": d["link_length"], "mean_tt": d["mean_tt"],
                                     "cv_tt": d["cv_tt"]} for d in data}
        self.link_info = {}
        for i, info in link_info_dict.items():
            length = info["length"]
            mean_speed = length / info["mean_tt"]
            cv = info["cv_tt"]
            self.link_info[i] = {"start_loc": start_milepost[i], "end_loc": end_milepost[i],
                                 "length": length, "mean_speed": mean_speed, "cv": cv}

    def get_line_info(self):
        stop_next_link = {s: s for s in range(self.stop_num)}
        link_next_stop = {l: l+1 for l in range(self.link_num-1)}
        link_next_stop[self.link_num-1] = 0

        init_link_dict = {}
        for init_link in self.bus_init_loc:
            visit_stops = list(range(init_link, self.stop_num)) + list(range(0, init_link))
            stop_depar_times = {}
            cum_time = 0.0
            for stop in visit_stops:
                stop_depar_times[stop] = cum_time
                link_idx = stop
                link_time = self.link_length[link_idx] / self.mean_speed
                beta = self.stop_beta[stop]
                dwell_time = beta * self.equ_headw
                cum_time += link_time + dwell_time

            init_link_dict[init_link] =stop_depar_times

        stop_depar_times = defaultdict(list)
        for init_link, stop_depar_time in init_link_dict.items():
            for s, t in stop_depar_time.items():
                stop_depar_times[s].append(t)

        stop_sched_times = defaultdict(list)
        for s, depar_times in stop_depar_times.items():
            min_depar_time = min(depar_times)
            stop_sched_times[s].append(min_depar_time)

        for s, sched_times in stop_sched_times.items():
            first_sched_time = sched_times[0]
            # adding 3 for safety
            count = int(self.sim_duration // self.equ_headw) + 3
            for i in range(count):
                sched_time = self.equ_headw * (i + 1) + first_sched_time
                sched_times.append(sched_time)

        for s, sched_times in stop_sched_times.items():
            if sched_times[0] == 0:
                sched_times.pop(0)

        self.line_info = {"bus_num": self.bus_num, "stop_num": self.stop_num, "link_next_stop": link_next_stop,
                          "stop_next_link": stop_next_link, "board_rate": board_rate, "alight_rate": alight_rate,
                          "bus_init_loc": self.bus_init_loc_dict, "capacity": 80, "stop_sched_times": stop_sched_times, "stop_beta": self.stop_beta}

    def get_equ_headway(self):
        H = Symbol('H')
        avg_trip_time = 0
        for link_len in self.link_length:
            link_time = link_len / self.mean_speed
            avg_trip_time += link_time
        for stop, info in self.stop_info.items():
            beta = self.stop_beta[stop]
            avg_trip_time += beta * H
        self.equ_headw = nsolve(avg_trip_time / self.bus_num - H, H, 1)
        self.avg_trip_time = avg_trip_time.subs(H, self.equ_headw)

    def get_stop_beta(self):
        self.stop_beta = {}
        for stop_seq, info in self.stop_info.items():
            arrive_rate_in_second = info["pax_arriv_rate"] / 60.0
            beta = arrive_rate_in_second / board_rate
            self.stop_beta[stop_seq] = beta


if __name__ == "__main__":
    config = Config(34)



