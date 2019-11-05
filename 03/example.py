import datetime

print("Start program")
time = datetime.datetime.now()
time_str = time.strftime("%Y/%m/%d %H:%M:%S")
print("現在時刻 = " + time_str)