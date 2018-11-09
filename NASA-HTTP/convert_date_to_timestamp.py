import time
from datetime import datetime

sample='01/Aug/1995:00:00:07'.replace('/','-').replace('Aug','08')
sample2='01/Aug/1995:01:02:07'.replace('/','-').replace('Aug','08')
d1=datetime.strptime(sample, '%d-%m-%Y:%H:%M:%S')
d2=datetime.strptime(sample2, '%d-%m-%Y:%H:%M:%S')
print(d1.timestamp(),d2.timestamp())
td=d2-d1
td_mins = int(round(td.total_seconds() / 60))
print(td_mins)
