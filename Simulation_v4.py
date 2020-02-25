# -*- coding: utf-8 -*-
"""
Created on Mon Dec 16 18:39:33 2019

@author: User
"""
#import numpy as np
import math
import random as rd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import copy


# parking lot  C/D Scheduling

np.set_printoptions(precision = 4)

class Simulation:

    def __init__(self, n_EV, n_Day):
        self.num_EV = n_EV
        self.num_Day = n_Day+1
        self.SlotPerHour = 4
        self.SlotTimeGap = 60/self.SlotPerHour
        self.num_TimeSlot = self.SlotPerHour * 24 * self.num_Day
        self.TotalSimTime = n_Day*24*60
        self.charging_power = 3.6
        self.discharging_power = 3.6

        self.charging_effi = 0.9
        self.discharging_effi = 0.9
        

        self.entry_EV = []
        self.entry_EV_Charging = {}
        self.entry_EV_Stay = []
        self.entry_EV_Depart = []
#        self.charging_load_list = [ 0 for _ in range(self.num_TimeSlot)]
        self.charging_load_list_ev = np.zeros(96)
        self.charging_load_list_grid = np.zeros(96)
        self.discharging_load_list_ev = np.zeros(96)
        self.discharging_load_list_grid = np.zeros(96)
        self.baseload = []
        self.basecost = []
        # 도착하자마자 전체 스케줄링하는 것... 스캐줄링의 단점은 베이스로드가 예측값이라는 것.
        
        in_filename = 'data/baseload.txt'
        fi = open(in_filename, 'r')
        lines  = fi.readlines()
        baseload = []
        tmp = []

        for line in lines:
            tmp.clear()
            items = line.split('\t') 
            for item in items:
                tmp.append(float(item))
            baseload.append(copy.deepcopy(tmp))
        self.np_baseload = np.array(baseload)


    def sim_init(self, day):
        self.entry_EV.clear()
        self.entry_EV_Charging.clear()
        self.entry_EV_Stay.clear()
        self.entry_EV_Depart.clear()
        self.charging_load_list_ev = np.zeros(96)
        self.charging_load_list_grid = np.zeros(96)
        self.discharging_load_list_ev = np.zeros(96)
        self.discharging_load_list_grid = np.zeros(96)
        self.baseload =  self.np_baseload[day]
        self.basecost = self.baseload

        # np.random.seed(100)
        arriv = np.random.normal(510, 60, self.num_EV)
        # count, bins, ignored = plt.hist(arriv, 30)
        # np.random.seed(200)
        stay = np.random.normal(600, 60, self.num_EV)
        # count, bins, ignored = plt.hist(stay, 30)
        # np.random.seed(200)
        soc = np.random.normal(0.6, 0.2, self.num_EV)
        for i in range(len(soc)):
            if soc[i] < 0:
                soc[i] = 0
            elif soc[i] >= 1.0:
                soc[i] = 0.95
        depart = arriv+stay
        for i in range(self.num_EV):
            self.entry_EV.append(self.sim_make_EV(i, 0, soc[i], arriv[i], stay[i], depart[i]))

        # sim.entry_EV.sort(key=lambda object: object.ID)
        # asoclist = []
        # for ev in range(len(sim.entry_EV)):
        #     asoclist.append(sim.entry_EV[ev].SoC)
        # print(asoclist)
        # plt.plot(asoclist)

        self.entry_EV.sort(key=lambda object:object.T_Arrive)


        arriv = np.random.normal(510, 60)
        stay = np.random.normal(600, 60)
        depart = arriv + stay
        soc = np.random.normal(0.6, 0.2)
        if soc < 0.1:
            soc = 0.1
        elif soc >= 1.0:
            soc = 0.95
        ev = self.sim_make_EV(0, 0, soc, arriv, stay, depart)
        return ev

    def sim_init_test(self, day):
        self.entry_EV.clear()
        self.entry_EV_Charging.clear()
        self.entry_EV_Stay.clear()
        self.entry_EV_Depart.clear()
        self.charging_load_list_ev = np.zeros(96)
        self.charging_load_list_grid = np.zeros(96)
        self.discharging_load_list_ev = np.zeros(96)
        self.discharging_load_list_grid = np.zeros(96)
        self.baseload = self.np_baseload[day]
        self.basecost = self.baseload

        # np.random.seed(100)
        arriv = np.random.normal(510, 60, self.num_EV)
        # count, bins, ignored = plt.hist(arriv, 30)
        # np.random.seed(200)
        stay = np.random.normal(600, 60, self.num_EV)
        # count, bins, ignored = plt.hist(stay, 30)
        # np.random.seed(200)
        soc = np.random.normal(0.6, 0.2, self.num_EV)
        for i in range(len(soc)):
            if soc[i] < 0:
                soc[i] = 0
            elif soc[i] >= 1.0:
                soc[i] = 0.95
        depart = arriv + stay
        for i in range(self.num_EV):
            self.entry_EV.append(self.sim_make_EV(i, 0, soc[i], arriv[i], stay[i], depart[i]))

        # sim.entry_EV.sort(key=lambda object: object.ID)
        # asoclist = []
        # for ev in range(len(sim.entry_EV)):
        #     asoclist.append(sim.entry_EV[ev].SoC)
        # print(asoclist)
        # plt.plot(asoclist)

        self.entry_EV.sort(key=lambda object: object.T_Arrive)


        return self.entry_EV

    def sim_step(self, action, ev, ts):
        loadstate = self.sim_get_load_state(ts)
        cost = 0
        reward = 0
        if action == 0:
            amount = -1*self.sim_Charging_EV(ts, ev)
            if amount == 0:
                reward = -1
            elif loadstate == 0:  # off
                reward = 3
            elif loadstate == 1:  # mid
                reward = 2
            elif loadstate == 2:  # peak
                reward = 1
        elif action == 1:
            amount = -1*self.sim_Discharging_EV(ts, ev)
            if loadstate == 0:  # off
                reward = 1
            elif loadstate == 1:  # mid
                reward = 2
            elif loadstate == 2:  # peak
                reward = 3
        elif action == 2:
            amount = self.sim_Idle_EV(ts, ev)


        cost = amount * self.basecost[ts]

        # loadstate = self.sim_get_peak_price_at_ts(ts)
        # if loadstate == 0:#off
        #     cost = amount*49.8
        # elif loadstate == 1:#mid
        #     cost = amount*94.2
        # elif loadstate == 2:#peak
        #     cost = amount*160.4
        # 충전은 +, 방전은 -

        done = self.sim_depart_check_EV(ts, ev)
        curload = self.baseload[ts]
        return np.array([ts, ev.SoC*100, ev.TS_Arrive, ev.TS_Depart, curload, loadstate]), reward, done, cost, amount*(-1)

    def sim_make_EV(self, id_EV, type_EV, soc, t_arr, t_stay, t_depart):
        ev = PEV(id_EV, type_EV, soc, t_arr, t_stay, t_depart)
        return ev
    def sim_slot_to_time(self, ts):
        day = math.floor(ts / (self.SlotPerHour * 24))
        hh = math.floor((ts % (self.SlotPerHour * 24)) / self.SlotPerHour)
        mm = math.floor(60 * ((ts % (self.SlotPerHour * 24)) % self.SlotPerHour) / self.SlotPerHour)
        return day, hh, mm
    def sim_time_to_slot_dhm(self, day, h, m):
        ts = day * (self.SlotPerHour * 24)
        ts = ts + h * self.SlotPerHour
        ts = ts + math.floor(m / (60 / self.SlotPerHour))
        return ts
    def sim_time_to_slot_m(self, m):
        ts = math.floor(m / (60 / self.SlotPerHour))
        return ts
    def sim_print_charging_list(self):
        for key in self.entry_EV_Charging:
            print('\n\n############################', key, self.sim_slot_to_time(key), '##########################')
            item = self.entry_EV_Charging[key]
            #            print(item)
            for i in range(len(item)):
                print(item[i].ID, end=' ')
    def sim_get_peak_price_at_ts(self, ts):

        if 92 <= ts < 96 or 0 <= ts < 36:
            # print("off")
            load_state = 0
        elif 36 <= ts < 40 or 48 <= ts < 68 or 80 <= ts < 88:
            # print('mid')
            load_state = 1
        elif 40 <= ts < 48 or 68 <= ts < 80 or 88 <= ts < 92:
            # print('peak')
            load_state =2

        return load_state

    def sim_get_load_state(self, ts):
        maxload = np.max(self.baseload)
        minload = np.min(self.baseload)
        midload = (maxload - minload)/2 + minload


        if minload <= self.baseload[ts] < (midload - minload)/2 + minload:
            # print("off")
            load_state = 0
        elif (midload - minload)/2 + minload <=self.baseload[ts] < (maxload - midload)/2 + midload:
            # print('mid')
            load_state = 1
        elif (maxload - midload)/2 + midload <= self.baseload[ts] <= maxload:
            # print('peak')
            load_state =2
        else:
            print('error')

            print( '{} {} {}'.format(minload, self.baseload[ts] , (midload - minload)/2 + minload ))
            print('{} {} {}'.format((midload - minload)/2 + minload , self.baseload[ts], (maxload - midload)/2 + midload))
            print('{} {} {}'.format((maxload - midload)/2 + midload, self.baseload[ts], maxload))
            load_state=0

        return load_state

    def sim_check_EVs(self, ts):
        # print(ts)
        e = 0
        while e < len(self.entry_EV):
            ev = self.entry_EV[e]
            if ts == ev.TS_Arrive:
                self.entry_EV.pop(e)
                self.entry_EV_Stay.append(ev)
                # ev.get_info_EV()
            else:
                break

    def sim_depart_check_EVs(self, ev, ts, done):


        idx = self.entry_EV_Stay.index(ev)
        print('*************  start num', len(self.entry_EV_Stay))
        print('idx', idx)
        print(self.entry_EV_Stay[idx].get_info_EV())
        if done !=  0:
            dev = self.entry_EV_Stay.pop(idx)
            self.entry_EV_Depart.append(dev)
            # print(dev.get_info_EV())
            print('======  depart  =======')
            print(dev.get_info_EV())
        else:
            print('no depart')


        print('**************** end num', len(self.entry_EV_Stay))

        return

    def sim_depart_check_EV(self, ts, dev):

        if ts == dev.TS_Depart-1:# 최종 결과가
            if dev.SoC < 0.8:
                done = -1
            else:
                done = 1
        elif dev.SoC == 0.: # 중간 결과가 0.2 보다 작으면 멈춰
            done = -2
        else:
            done = 0

        return done

    def sim_Charging_EV(self, ts,  ev):
        value = ev.pev_soc_update(self.charging_power/self.SlotPerHour)
        self.charging_load_list_ev[ts] = self.charging_load_list_ev[ts] + value
        self.charging_load_list_grid[ts] = self.charging_load_list_grid[ts] + value
        return value
        # print('sim charging ev', (value/(self.charging_effi)))
    def sim_Discharging_EV(self, ts,  ev):
        value = ev.pev_soc_update(-(self.discharging_power/self.SlotPerHour))
        self.discharging_load_list_ev[ts] = self.discharging_load_list_ev[ts] + value
        self.discharging_load_list_grid[ts] = self.discharging_load_list_grid[ts] + value*self.discharging_effi
        return value
        # print('sim discharging ev', (value/(self.discharging_effi)))
    def sim_Idle_EV(self, ts,  ev):
        value = ev.pev_soc_update(0)
        # print('sim idle ev')
        # print(value)
        return 0

    def sim_scheduling_discrete_less_load(self, ev): #

        req_power = ev.req_battery_charging*(2-self.charging_effi)
        req_timeslot = req_power/(self.charging_power/self.SlotPerHour)
        remain = req_power%(self.charging_power/self.SlotPerHour)
        req_charging_timeslot = math.ceil(req_timeslot)

        avail_num_charging = ev.TS_Depart - ev.TS_Arrive-1
        start_slot = ev.TS_Arrive+1
        end_slot = ev.TS_Depart-1
        charging_block_list = []

        if req_charging_timeslot > 0 and avail_num_charging >=  req_charging_timeslot:
            for s in range(avail_num_charging):
                slot_load = 0
                slot_load = self.charging_load_list[s+start_slot] + self.baseload[s+start_slot]
                charging_block_list.append((s+start_slot, slot_load))

            charging_block_list.sort(key=lambda element:element[1])

            for k in range(req_charging_timeslot):
                s, v = charging_block_list.pop(0)

                if k == req_charging_timeslot-1:
                    self.charging_load_list[s] += remain
                else:
                    self.charging_load_list[s] += self.charging_power/self.SlotPerHour

        return 0


class PEV:
    def __init__(self, id_EV, typeEV, soc, t_arr, t_stay, t_depart):
        self.ID = id_EV
        self.ChargingLimit = 1.0
        self.SoC = soc
        self.init_SoC = soc

#        self.SoC = 0.6
        self.charging_effi = 0.9
        self.charging_power = 3.6
        self.SlotPerHour = 4

        self.type_EV = typeEV
        if typeEV == 0:
            self.battery_capa = 16.4
        elif typeEV == 1:
            self.battery_capa = 27
        elif typeEV == 2:
            self.battery_capa = 30
        elif typeEV == 3:
            self.battery_capa = 40
            
        if self.ChargingLimit - self.SoC < 0:
            self.req_SoC = 0
        else:
            self.req_SoC = self.ChargingLimit - self.SoC

        self.cur_bat_power = self.SoC*self.battery_capa

        # self.req_battery_charging = self.req_SoC * self.battery_capa
        self.T_Arrive = t_arr
        self.T_Stay = t_stay
        self.T_Depart = t_depart
        self.TS_Arrive = self.sim_time_to_slot_m_ceil(t_arr)
        self.TS_Depart = self.sim_time_to_slot_m_floor(t_depart)
        self.TS_Stay = self.TS_Depart - self.TS_Arrive

        self.req_bat_power = self.battery_capa - self.cur_bat_power
        # req_timeslot = self.req_power / (self.charging_power / self.SlotPerHour)
        # self.remain = self.req_power % (self.charging_power / self.SlotPerHour)
        # self.req_charging_timeslot = math.ceil(req_timeslot)

    def pev_soc_update(self, value):

        # print(value)

        result_value = 0.0
        if value > 0: #charging

            if self.req_bat_power >= value*self.charging_effi:
                self.cur_bat_power = self.cur_bat_power +  value * self.charging_effi
                result_value = value
            else:
                self.cur_bat_power = self.cur_bat_power + self.req_bat_power
                result_value = self.req_bat_power/self.charging_effi

        elif value < 0 :#discharging
            if self.cur_bat_power + value >= 0:
                self.cur_bat_power = self.cur_bat_power + value
                result_value = value
            else:
                self.cur_bat_power = 0
                result_value = -self.cur_bat_power
        else:
            result_value = 0.0

        self.SoC = self.cur_bat_power/self.battery_capa
        self.req_bat_power = self.battery_capa - self.cur_bat_power

        return result_value

    def sim_time_to_slot_m_floor(self, m):
        ts = math.floor(m/(60/self.SlotPerHour))
        return ts
    def sim_time_to_slot_m_ceil(self, m):
        ts = math.ceil(m/(60/self.SlotPerHour))
        return ts
    def get_SoC(self):
        return self.SoC
    def get_info_EV(self):
        print('ID: '+str(self.ID))
        print('EV-Type: ' + str(self.type_EV)+'    SoC: ' + str(self.SoC)+'    battery_capa: ' + str(self.battery_capa)+'    cur_bat_power: '+str(self.cur_bat_power))
        print('TS_Arr: ' + str(self.TS_Arrive)+'    TS_Stay: ' + str(self.TS_Stay)+'    TS_Depart: ' + str(self.TS_Depart))
        # print('T_Arr: ' + str(self.T_Arrive)+'    T_Stay: ' + str(self.T_Stay)+'    T_Depart: ' + str(self.T_Depart))
    def get_str_info_EV(self):
        s = '  =>>  ID:'+str(self.ID)+' EV-Type:' + str(self.type_EV)+' SoC: ' + str(self.SoC)+' Battery_capa: ' + str(self.battery_capa)+' Req_battery_charging: ' + str(self.req_battery_charging) + ' TS_Arr: ' + str(self.TS_Arrive)+' TS_Stay: ' + str(self.TS_Stay)+' TS_Depart: ' + str(self.TS_Depart)+' T_Arr: ' + str(self.T_Arrive)+' T_Stay: ' + str(self.T_Stay)+' T_Depart: ' + str(self.T_Depart)
        return s
    def get_TimeToSlot(self):
        self.TS_Depart = self.T_Depart
        self.TS_Stay =self.T_stay
        self.TS_Arrive = self.T_Arrive

if __name__ == "__main__":
    sim = Simulation(100, 7)

    for day in range(1):
        sim.sim_init(day)

        for t in range(96):
            sim.sim_check_EVs(t)
            # print(t, len(sim.entry_EV), len(sim.entry_EV_Stay), len(sim.entry_EV_Depart))
            for e in range(len(sim.entry_EV_Stay)):
                ev = sim.entry_EV_Stay[e]
                soc = ev.SoC
                # print(t, ev.ID, soc)
                action = np.random.randint(0, 3)
                # action = 1
                if action == 0:
                    sim.sim_Charging_EV(t, ev)
                elif action ==1:
                    sim.sim_Discharging_EV(t, ev)
                else:
                    sim.sim_Idle_EV(t, ev)

            # sim.sim_after_check_EVs(t)

        sim.entry_EV_Depart.sort(key=lambda object: object.ID)
        soclist = []
        for ev in range(len(sim.entry_EV_Depart)):
            # print(sim.entry_EV_Depart[ev].ID, sim.entry_EV_Depart[ev].SoC)
            soclist.append(sim.entry_EV_Depart[ev].SoC)
        print(np.average(soclist))
        plt.bar(np.arange(len(sim.entry_EV_Depart)), soclist)
        plt.show()

        print(sim.charging_load_list)
        print(sim.discharging_load_list)
        plt.bar(np.arange(96), sim.charging_load_list + sim.baseload)
        plt.bar(np.arange(96), sim.baseload)
        plt.bar(np.arange(96), sim.discharging_load_list)
        plt.ylim(-100, 220)
        plt.show()