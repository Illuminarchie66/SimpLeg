import json
import pickle
import os
import re
from Pipeline.LegislationPipe import LegislationPipe

class Legislation:
    def __init__(self, year, id):
        self.year = year 
        self.id = id

        self.rawText = None
        self.json = None 
        self.chunks = None
        self.summary = None
        self.metrics = None

        self.directory = "../Legislation/Leg-" + str(self.year) + "-" + str(self.id)

        self.written = False

    def printJSON(self):
        print(json.dumps(self.json, indent=4))

    def writeLeg(self):
        self.writeRaw()
        self.writeJSON()
        self.writeChunks()
        self.written = True
    
    def formatRaw(self):
        result = re.split(r'\n\n', self.rawText)
        resultm = [string.replace('\n', '') for string in result]
        w = ', \n'.join(resultm)
        return w

    def writeRaw(self):
        if self.rawText != None:
            w = self.formatRaw()
            with open(f"{self.directory}/raw-{self.year}-{self.id}.txt", "w") as file:
                file.write(w)

    def formatJSON(self):
        return json.dumps(self.json, indent=4)

    def writeJSON(self):
        if self.json != None:
            w = self.formatJSON()
            with open(f"{self.directory}/json-{self.year}-{self.id}.json", "w") as file:
                file.write(w)

    def formatChunks(self):
        text = ""
        for chunk in self.chunks:
            w = "\n---------------------------------------------" + chunk['text'] + "\n---------------------------------------------\n"
            text += w
        return text

    def writeChunks(self):
        if self.chunks != None:
            w = json.dumps(self.chunks, indent=4)
            with open(f"{self.directory}/chunked-{self.year}-{self.id}.json", "w") as file:
                file.write(w)

    def formatSummary(self):
        ret = "{\n\t \"general\": " + self.summary['general']
        ret += "\n\t \"series\": ["
        for res in self.summary['series']:
            ret += "\n\t\t ["
            for r in res: 
                ret += "\n\t\t\t {"
                ret += "\n\t\t\t\t \"length\": " + str(r['length'])
                ret += "\n\t\t\t\t \"short\": " + r['short']
                ret += "\n\t\t\t\t \"text\": "
                for c in r['text'].split("\n"):
                    ret += "\n\t\t\t\t\t" + c
                ret += "\n\t\t\t },"
            ret = ret[:-1]
            ret += "\n\t\t ],"
        ret = ret[:-1]
        ret += "\n\t ]"
        ret += "\n\t \"summary\": " + self.summary['summary']
        ret += "\n}"
        return ret

    def writeSummary(self):
        if self.summary != None:
            w = json.dumps(self.summary, indent=4)
            with open(f"{self.directory}/summ-{self.year}-{self.id}.json", "w") as file:
                file.write(w)

    def formatMetrics(self):
        return str(self.metrics)

    def writeMetrics(self):
        if self.metrics != None:
            with open(f'{self.directory}/metrics-{self.year}-{self.id}.pkl', 'wb') as file:
                pickle.dump(self.metrics, file)
            # w = json.dumps(vars(self.metrics), indent=4, default=lambda o: o.__dict__)
            # with open(f"{self.directory}/metrics-{self.year}-{self.id}.json", "w") as file:
            #     file.write(w)

