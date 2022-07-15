#!/usr/local/bin/python3
import os
import json
import psutil
import psycopg2 as psy
from datetime import timedelta
from datetime import datetime as dt
from psycopg2.extras import execute_values

class DataCollection:
    def __init__(self):
        self.con = psy.connect(host='localhost', port=5432, dbname='postgres')
        self.crsr = self.con.cursor()

    def get_data(self):
        date_now = dt.now().strftime('%Y-%m-%d %H:%M:%S')
        cpu_use = psutil.cpu_percent()
        mem_use = getattr(psutil.virtual_memory(), 'percent')
        swap_use = getattr(psutil.swap_memory(), 'percent')
        disk_use = getattr(psutil.disk_usage('/'), 'percent')
        battery_charge = getattr(psutil.sensors_battery(), 'percent')
        x = (date_now, cpu_use, mem_use, swap_use, disk_use, battery_charge)
        return x

    def insert_data(self, lst):
        create_sql = """
        create table if not exists laptop_info
        (
            date_now date not null,
            cpu_use float,
            mem_use float,
            swap_use float,
            disk_use float,
            battery_charge float
        );
        """
        self.crsr.execute(create_sql)
        with self.con, self.crsr:
            self.crsr.execute("""Insert into postgres.public.laptop_info (date_now, cpu_use, mem_use, swap_use,
             disk_use,battery_charge) values ( %s, %s, %s, %s, %s, %s )""", lst)

        # check_res = '''select count(*) from postgres.public.laptop_info where date_now >
        #                 ('2022-07-14'::date)'''
        # self.crsr.execute(check_res)
        # res = self.crsr.fetchall()
        # print(res)


if __name__ == "__main__":
    dc = DataCollection()
    lst = dc.get_data()
    dc.insert_data(lst)
