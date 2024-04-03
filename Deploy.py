from Pipeline.Generator import Generator
from Pipeline.LegislationBuilder import LegislationBuilder
import json
from Pipeline.TextMetrics import TextMetrics, TextMetricBuilder
import time
import random
import pandas as pd

generator = Generator(verbose=True)
LegBuilder = LegislationBuilder(generator.chainBuilder.tokenizer)
metricBuilder = TextMetricBuilder()

#bills = [(2019, 5), (2020, 6), (2023, 5), (2023, 13), (2023, 26)
years = range(2000, 2019)
ids = range(1,51)
bills = [(year, id) for year in years for id in ids]
random.shuffle(bills)
#bills = [(2018,30), (2019, 10), (2020, 14), (2021, 2), (2022, 1)]
data = []
for bill in bills:
    try:
        startTime = time.time()
        leg = LegBuilder(bill[0], bill[1], preFile=True, verbose=False)
        leg.writeRaw()
        leg.writeJSON()
        print(leg.link)

        if leg.json['wordSum'] <= 60000:

            if leg.summary == None:
                response = generator.generate(legislation=leg)
                leg.summary = response
                leg.writeSummary()

            if leg.metrics == None:
                metrics = metricBuilder(leg) 
                leg.metrics = metrics
                leg.writeMetrics()
            
            t = time.time() - startTime
            d = leg.metrics.to_dict()
            d['year'] = bill[0]
            d['id'] = bill[1]
            d['time'] = t
            data.append(d)

            print(str(bill[0]) + "-" + str(bill[1]) +": success")
            df = pd.json_normalize(data, sep='-')
            df.to_csv('data.csv', index=False)
    except Exception as e:
       print(str(e))
       print(str(bill[0]) + "-" + str(bill[1]) +": fail")


