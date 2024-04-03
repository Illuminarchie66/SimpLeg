from flask import Flask, render_template, request
from Pipeline.LegislationBuilder import LegislationBuilder
from Pipeline.Legislation import Legislation
import os
import re
import json
import pickle
from Pipeline.Readability import ReadabilityDetails

legbuilder = LegislationBuilder(None)

def extract_info(string):
    parts = string.split('-')
    year = int(parts[1])
    identifier = int(parts[2])

    leg = legbuilder(year, identifier, preFile=True)
    directory = "Legislation/Leg-" + str(year) + "-" + str(identifier)

    def getGrade(readMetrics):
        read = ["flesch_kincaid_grade","smog","gunning_fog","automated_readability_index","coleman_liau_index"]
        g = 0
        for r in read:
            g += readMetrics[r]
        g = g/len(read)
        return ReadabilityDetails.averageToUK(g)
        

    if leg.metrics == None or leg.summary == None:
        return None
    else:
    
        result = {'year': year, 
                'id': identifier,
                'link': leg.link,
                'title': leg.title,
                'longtitle': leg.longtitle,
                'date': leg.date,
                'url':  f'legislation/year={year}&id={identifier}',
                'general': leg.summary['general'],
                'textmetrics': leg.metrics.to_dict(),
                'gradeOriginal': getGrade(leg.metrics.fullTextReadability),
                'gradeSummary': getGrade(leg.metrics.fullSumm0Readability)}
        return result

def get_directory_list():
    # Get the current working directory
    current_directory = 'Legislation'
    
    # Get a list of directories in the current directory
    directories = [d for d in os.listdir(current_directory) if os.path.isdir(os.path.join(current_directory, d))]
    dirs = [extract_info(d) for d in directories]
    dirs = [d for d in dirs if d is not None]
    return dirs

app = Flask(__name__)

@app.route('/')
def index():
    dirs = get_directory_list()
    return render_template('index.html', legislation=dirs)

@app.route('/legislation/year=<int:Year>&id=<int:Id>')
def legislation(Year, Id):
    leg = legbuilder(Year, Id, preFile=True, verbose=True)

    def process(text):
        text = text.lstrip("\n")
        return re.sub(r'\n', '<br>', text)
    
    def simpProcess(text):
        text = text.lstrip("\n")
        text = re.sub(r'\n+', '\n', text)
        matches = re.findall(r'Simplified point \d+: .*\.', text)
        return [re.search(r'Simplified point \d+: (.*)\.', m).group(1) + "." for m in matches]
    
    def removeQuotes(text):
        if text.startswith('"'):
            text = text[1:]  # Remove the leading quote
        if text.endswith('"'):
            text = text[:-1]  # Remove the trailing quote
        return text

    return render_template('legislation.html',
                           link = leg.link,
                           year = leg.year,
                           page = leg.id,
                           general = process(leg.summary['general']),
                           summary = process(leg.summary['summary']),
                           chunks = [{'text': simpProcess(c['text']), 'short': removeQuotes(process(c['short']))} for c in leg.summary['series'][0]],
                           title = leg.title,
                           longittle = leg.longtitle,
                           metrics = leg.metrics.to_dict())

if __name__ == '__main__':
    app.run(debug=True)
