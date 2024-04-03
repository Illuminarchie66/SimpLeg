from Pipeline.Legislation import Legislation
from Pipeline.LegislationPipe import LegislationPipe
import time
import pickle
import os
import json
import re
from Pipeline.TextMetrics import TextMetrics

class LegislationBuilder:

    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, year, id, preFile=False, webLoad=False, verbose=False):
        startTime = time.time()

        leg = Legislation(year, id)

        directory = "../Legislation/Leg-" + str(year) + "-" + str(id)
        jsonPath = f"{directory}/json-{year}-{id}.json"
        rawPath = f"{directory}/raw-{year}-{id}.txt"
        chunkPath = f"{directory}/chunked-{year}-{id}.json"
        summPath = f"{directory}/summ-{year}-{id}.json"
        metricPath = f"{directory}/metrics-{year}-{id}.pkl"
        if (preFile and os.path.exists(jsonPath) and os.path.exists(rawPath)):
            print("Preload")
            with open(f"{directory}/json-{year}-{id}.json", 'r') as f:
                data = json.load(f)
                leg.link = data['link']
                leg.title = data['title']
                leg.longtitle = data['longtitle']
                leg.date = data['date']
                leg.totalTokens = data['totalTokens']
                leg.json = data
                leg.written = True

            with open(f"{directory}/raw-{year}-{id}.txt", 'r', encoding="utf-8") as f:
                leg.rawText = f.read()
 
            if os.path.exists(chunkPath):
                with open(f"{directory}/chunked-{year}-{id}.json", 'r', encoding="utf-8") as f:
                    leg.chunks = json.load(f)

            if os.path.exists(summPath):
                with open(f"{directory}/summ-{year}-{id}.json", 'r', encoding="utf-8") as f:
                    leg.summary = json.load(f)

            if os.path.exists(metricPath):
                with open(f"{directory}/metrics-{year}-{id}.pkl", 'rb') as file:
                    leg.metrics = pickle.load(file)

            return leg
        
        leg.link = "https://www.legislation.gov.uk/ukpga/{year}/{id}/enacted?view=plain".format(year=year, id=id) 
        soup = LegislationPipe.getLegislationHTML(year, id)

        leg.title = LegislationPipe.getLegislationHTMLTitle(soup)
        leg.longtitle = LegislationPipe.getLegislationHTMLLongTitle(soup)
        leg.date = LegislationPipe.getLegislationHTMLDate(soup)

        leg.html = LegislationPipe.getLegislationHTMLBody(soup)
        print("HTML:\n--- %s seconds ---" % (time.time() - startTime)) if verbose else None
        startTime = time.time()
        leg.rawText = LegislationPipe.getLegislatonString(leg.html)
        print("Raw text:\n--- %s seconds ---" % (time.time() - startTime)) if verbose else None
        startTime = time.time()

        tree = LegislationPipe.getLegislationTree(leg.rawText, self.tokenizer)

        leg.json = LegislationPipe.getLegislationJSON(tree) 
        leg.json['link'] = leg.link
        leg.json['title'] = leg.title
        leg.json['longtitle'] = leg.longtitle
        leg.json['date'] = leg.date

        counts = LegislationPipe.setJSONLengths(leg.json, self.tokenizer, -3)
        print("JSON:\n--- %s seconds ---" % (time.time() - startTime)) if verbose else None
        startTime = time.time()
              
        leg.totalTokens = counts[1]
        leg.json['totalTokens'] = leg.totalTokens

        os.makedirs(directory, exist_ok=True)

        return leg
        