import datetime

def get_datetime(year, day, hour=0, minute=0, second=0):
    """ Returns month and day given a day of a year"""

    dt = datetime.datetime(year, 1, 1, hour, minute, second) + datetime.timedelta(days=day-1)
    return dt

def get_file_time_info(radiance_filename, split_char='MYD021KM.A'):

    time_info = radiance_filename.split(split_char)[1]
    year, abs_day = time_info[:4], time_info[4:7]
    hour, minute = time_info[8:10], time_info[10:12]

    return year, abs_day, hour, minute

def minutes_since(year, abs_day, hour, minute, ref_year=2008, ref_abs_day=1, ref_hour=0, ref_minute=0, ref_second=0):
    
    dt = get_datetime(year, abs_day, hour, minute)
    ref_dt = get_datetime(ref_year, ref_abs_day, ref_hour, ref_minute, ref_second)

    return int((dt - ref_dt).total_seconds() // 60)