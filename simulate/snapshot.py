class Snapshot:
    def __init__(self, episode, ct, curr_bus_id, curr_stop_id, buses, stops, config) -> None:
        '''
        Take a snapshot of the system and formulate different states based on agent type
        '''
        self.episode = episode
        self.ct = ct
        self.curr_bus_id = curr_bus_id
        self.curr_stop_id = curr_stop_id
        # get corridor length for loop corridor
        last_link_id = max(config.link_info.keys())
        self.corri_lengt = config.link_info[last_link_id]['end_loc']
        self.stop_num = config.stop_num
        self.bus_num = config.bus_num
        self.equ_headway = config.equ_headw

        local_state, globa_state, pos_idx, globa_relat_state = self.get_spacing_state(
            buses, curr_bus_id)
        # local state, forward and backward spacing
        self.local_state = local_state
        # all buses' abs locations
        self.globa_state = globa_state
        self.pos_idx = pos_idx
        # buses' relative location from the current bus
        self.globa_relat_state = globa_relat_state
        self.last_depar_times = stops[curr_stop_id].last_depar_times
        self.pax_demand = stops[curr_stop_id].pax_on_stop_num
        self.trip_no = buses[curr_bus_id].get_trip_no()
        self.bus_load = buses[curr_bus_id].get_onboard_num()

    def get_spacing_state(self, buses, curr_bus_id):
        '''
        Parameters:
            buses - dict, bus_id -> bus object
        '''
        # all_bus_loc_info = {b.bus_id: b.get_bus_loc_info() for b in buses}
        all_bus_x = {b.bus_id: b.get_bus_loc_info()['x'] for b in buses}
        # sorted [bus_id, x]
        sorte_all_bus_x = sorted(all_bus_x.items(), key=lambda x: x[1])
        bus_num = len(sorte_all_bus_x)
        # store global state (each bus's location)
        globa_xs = []
        # store local state (querying bus's spacing between the two neighbors)
        for_ward_x, back_ward_x = None, None
        # store querying bus's position (in integer)
        pos_idx = None
        # current querying bus's location in x
        curr_x = None
        for idx, (bus_id, x) in enumerate(sorte_all_bus_x):
            globa_xs.append(x)
            if bus_id != curr_bus_id:
                continue
            pos_idx = idx
            curr_x = x
            # querying bus is at the most downstream
            if idx == bus_num - 1:
                for_ward_x = (self.corri_lengt - x) + sorte_all_bus_x[0][1]
                back_ward_x = x - sorte_all_bus_x[idx - 1][1]
            # querying bus is at the most upstream
            elif idx == 0:
                for_ward_x = sorte_all_bus_x[idx + 1][1] - x
                back_ward_x = self.corri_lengt - sorte_all_bus_x[-1][1] + x
            # querying bus is at the middle
            else:
                for_ward_x = sorte_all_bus_x[idx + 1][1] - x
                back_ward_x = x - sorte_all_bus_x[idx - 1][1]
            assert for_ward_x >= 0 and back_ward_x >= 0, 'calculation error'

        local_state = [for_ward_x / self.corri_lengt,
                       back_ward_x / self.corri_lengt]     
        globa_state = [x / self.corri_lengt for x in globa_xs]    
        globa_relat_state = self.cal_relative_global_state(globa_xs, curr_x)   
        return local_state, globa_state, pos_idx, globa_relat_state

    def cal_relative_global_state(self, xs, x):
        '''
        get the relative location of all buses from the querying bus
        '''
        if self.bus_num % 2 == 0:
            curr_idx = xs.index(x)
            for_bus_num = self.bus_num // 2 - 1
            for_spacs = []
            for dt in range(1, for_bus_num + 1, 1):
                idx = curr_idx + dt
                if idx <= (self.bus_num - 1):
                    for_spacs.append(xs[idx] - x)
                else:
                    for_spacs.append(
                        xs[idx % self.bus_num] + self.corri_lengt - x)

            back_bus_num = self.bus_num // 2
            back_spacs = []
            for dt in range(1, back_bus_num + 1, 1):
                idx = curr_idx - dt
                if idx >= 0:
                    back_spacs.append((x - xs[idx]))
                else:
                    back_spacs.append(
                        (self.corri_lengt - xs[idx % self.bus_num] + x))

            spacs = sorted(for_spacs, reverse=True) + back_spacs
            spacs = [x / self.corri_lengt for x in spacs]
            return spacs
        else:
            curr_idx = xs.index(x)
            for_bus_num = self.bus_num // 2
            for_spacs = []
            for dt in range(1, for_bus_num + 1, 1):
                idx = curr_idx + dt
                if idx <= (self.bus_num - 1):
                    for_spacs.append(xs[idx] - x)
                else:
                    for_spacs.append(
                        xs[idx % self.bus_num] + self.corri_lengt - x)

            back_bus_num = self.bus_num // 2
            back_spacs = []
            for dt in range(1, back_bus_num + 1, 1):
                idx = curr_idx - dt
                if idx >= 0:
                    back_spacs.append((x - xs[idx]))
                else:
                    back_spacs.append(
                        (self.corri_lengt - xs[idx % self.bus_num] + x))

            spacs = sorted(for_spacs, reverse=True) + back_spacs
            # spacs = spacs[for_bus_num-1: for_bus_num+1]
            spacs = [x / self.corri_lengt for x in spacs]
            return spacs
