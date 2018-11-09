import time
from datetime import datetime

sample='18-Nov-1994:09:58:56'.replace('/','-')
sample=sample.replace(str(sample[3:6]),'08')
sample2='01/Aug/1995:01:02:07'.replace('/','-')
sample2=sample2.replace(str(sample2[3:6]),'08')
d1=datetime.strptime(sample, '%d-%m-%Y:%H:%M:%S')
d2=datetime.strptime(sample2, '%d-%m-%Y:%H:%M:%S')
print(d1.timestamp(),d2.timestamp())
td=d2-d1
td_mins = int(round(td.total_seconds() / 60))
print(td_mins)
