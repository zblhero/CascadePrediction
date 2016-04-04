#coding = utf-8

import sys
reload( sys )
sys.setdefaultencoding('utf-8')

import re, time
from datetime import datetime
from datetime import timedelta

class WTime:

    def __init__(self, string):
        self.string = string.decode('utf8')
        self.to_standard()

    def to_standard(self):
        month_pattern = re.compile(ur'[0-9]+月')
        if re.search(ur'月', self.string):
            year = 2015
            month = int(re.search(ur'[0-9]+月', self.string).group()[:-1])
            day = int(re.search(ur'[0-9]+日', self.string).group()[:-1])

            hour = int(re.search(ur'[0-9]+:', self.string).group()[:-1])
            minute = int(re.search(ur':[0-9]+', self.string).group()[1:])
            second = 0
            
            self.dt = datetime(year, month, day, hour, minute, second)
        elif re.search(ur'今天', self.string):

            hour = int(re.search(ur'[0-9]+:', self.string).group()[:-1])
            minute = int(re.search(ur':[0-9]+', self.string).group()[1:])
            second = 0

            today = datetime.today()
            self.dt = datetime(today.year, today.month, today.day, hour, minute, second)
        elif re.search(ur'分钟前', self.string):
            past_min = int(re.search(ur'[0-9]+', self.string).group())
            time_delta = timedelta(minutes = past_min)
            cur_mt =  datetime.now()

            self.dt = cur_mt - time_delta
        elif re.search(ur'刚刚', self.string):
            self.dt = datetime.now()
        else:
            self.dt = datetime.strptime(self.string, '%Y-%m-%d %H:%M:%S')

        self.st = datetime.strftime(self.dt, '%Y-%m-%d %H:%M:%S')
        self.mt = time.mktime(self.dt.timetuple())


