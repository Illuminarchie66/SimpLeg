from rouge_score import rouge_scorer
from nltk.translate.bleu_score import corpus_bleu
from nltk.tokenize import word_tokenize
from evaluate import load
from language_tool_python import LanguageTool, Match
import nltk
import re
import spacy
import textdescriptives as td
import pandas as pd

from transformers import pipeline

class TextMetricBuilder:
    def __init__(self):
        self.rougeScorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL', 'rougeLsum'], use_stemmer=True)
        self.bertscore = load("bertscore")
        self.tool = LanguageTool('en-US')
        self.sentimentPipeline = pipeline(model="ProsusAI/finbert")
        self.nlp = spacy.load("en_core_web_sm")
        self.nlp.add_pipe("textdescriptives/descriptive_stats")
        self.nlp.add_pipe("textdescriptives/readability")
        self.nlp.add_pipe("textdescriptives/coherence")
        self.nlp.add_pipe("textdescriptives/information_theory")

    def rougeScore(self, original, summary):
        return self.rougeScorer.score(original, summary)

    def bleuScore(self, original, summary, n=5):
        originalTokens = word_tokenize(original.lower())
        summTokens = word_tokenize(summary.lower())
        originalNgrams = [nltk.ngrams(originalTokens, i) for i in range(1, n)]
        originalNgrams = [list(ngram) for ngram in originalNgrams]
        summNgrams = [nltk.ngrams(summTokens, i) for i in range(1, n)]
        summNgrams = [list(ngram) for ngram in summNgrams]

        return corpus_bleu(originalNgrams, summNgrams)

    def bertScore(self, original, summary):
        return self.bertscore.compute(references=[original], predictions=[summary], model_type="distilbert-base-uncased")

    def textPreprocess(self, text):
        text2 = re.sub(r'\n+', '', text)
        text2 = re.sub(r'\t+', '', text2)
        return text2
    
    def textStats(self, doc):
        stats = doc._.descriptive_stats
        stats['n_syllables'] = sum(doc._._n_syllables)
        letters = 0 
        for token in doc:
            if token.is_alpha:
                letters += len(str(token))
        stats['n_letters'] = letters
        return stats 
    
    def infoTheory(self, doc):
        info = {'perplexity': doc._.perplexity, 'entropy': doc._.entropy, 'coherence': doc._.coherence}
        return info

    def __call__(self, legislation):
        # -------- Preprocessing --------
        metrics = TextMetrics()

        metrics.jsonPreprocess(legislation.json)
        fullText = self.textPreprocess(metrics.fullText)

        chunks = []
        for c in legislation.chunks:
            chunks.append(self.textPreprocess(c['text']))
        
        general = self.textPreprocess(legislation.summary['general'])
        summary = self.textPreprocess(legislation.summary['summary'])
        series = []
        for iteration in legislation.summary['series']:
            i = []
            for s in iteration:
                i.append(self.textPreprocess(s['short'] + "\n" + s['text']))
            series.append(i)
        
        fullSumm0 = general + "\n"
        for s in series[0]:
            fullSumm0 += s +"\n"
        fullSumm0 += summary 
        fullSumm0 = self.textPreprocess(fullSumm0)

        if len(series) > 1:
            fullSummN = general + "\n"
            for s in series[len(series)-1]:
                fullSummN += s +"\n"
            fullSummN += summary 
            fullSummN = self.textPreprocess(fullSummN)
        else:
            fullSummN = None
        
        partSumm = self.textPreprocess(general + "\n" + summary)

        fullTextNLP = self.nlp(fullText)
        chunksNLP = [self.nlp(s) for s in chunks]
        generalNLP = self.nlp(general)
        series0NLP = [self.nlp(s) for s in series[0]]
        summNLP = self.nlp(summary)
        fullSumm0NLP = self.nlp(fullSumm0)
        fullSummNNLP = self.nlp(fullSummN) if fullSummN != None else None
        partSummNLP = self.nlp(partSumm)

        # -------- Text metrics --------
        metrics.fullTextStats = self.textStats(fullTextNLP)
        metrics.chunksStats = [self.textStats(c) for c in chunksNLP]
        metrics.generalStats = self.textStats(generalNLP)
        metrics.series0Stats = [self.textStats(s) for s in series0NLP]
        metrics.summStats = self.textStats(summNLP)
        metrics.fullSumm0Stats = self.textStats(fullSumm0NLP)
        metrics.fullSummNStats = self.textStats(fullSummNNLP) if fullSummNNLP != None else None
        metrics.partSummStats = self.textStats(partSummNLP)

        # -------- Readability metrics --------
        metrics.fullTextReadability = fullTextNLP._.readability
        metrics.chunksReadability = [c._.readability for c in chunksNLP]
        metrics.generalReadability = generalNLP._.readability
        metrics.series0Readability = [s._.readability for s in series0NLP]
        metrics.summReadability = summNLP._.readability
        metrics.fullSumm0Readability = fullSumm0NLP._.readability
        metrics.fullSummNReadability = fullSummNNLP._.readability if fullSummNNLP != None else None
        metrics.partSummReadability = partSummNLP._.readability

        # -------- Summarisation metrics --------
        def getfmeasure(dictionary):
            new_dict = {}
            for key, value in dictionary.items():
                if len(value) >= 3:  # Ensure the list has at least three elements
                    new_dict[key] = value[2]
            return new_dict

        metrics.chunkSimpRouge = [getfmeasure(self.rougeScore(c,s)) for c,s in zip(chunks, series[0])]
        metrics.fullPartSummRouge = getfmeasure(self.rougeScore(fullText, partSumm))
        metrics.full2Summ0Rouge = getfmeasure(self.rougeScore(fullText, fullSumm0))
        metrics.full2SummNRouge = getfmeasure(self.rougeScore(fullText, fullSummN)) if fullSummN != None else None

        metrics.chunkSimpBleu = [self.bleuScore(c,s) for c,s in zip(chunks,series[0])]
        metrics.fullPartSummBleu = self.bleuScore(fullText, partSumm) 
        metrics.full2Summ0Bleu = self.bleuScore(fullText, fullSumm0)
        metrics.full2SummNBleu = self.bleuScore(fullText, fullSummN) if fullSummN != None else None

        metrics.chunkSimpBert = [self.bertScore(c,s) for c,s in zip(chunks,series[0])]
        metrics.fullPartSummBert = self.bertScore(fullText, partSumm) 
        metrics.full2Summ0Bert = self.bertScore(fullText, fullSumm0)
        metrics.full2SummNBert = self.bertScore(fullText, fullSummN) if fullSummN != None else None

        # -------- Orthographic metrics --------
        matches = self.tool.check(metrics.fullText)
        matches = [m for m in matches if m.ruleId != "MORFOLOGIK_RULE_EN_US" ]
        metrics.originalMatches = matches

        spellSummary = legislation.summary['general'] + "\n" + legislation.summary['summary'] + "\n"
        for s in legislation.summary['series'][0]:
            spellSummary += s['short'] + "\n" + s['text'] + "\n"

        matches = self.tool.check(spellSummary)
        matches = [m for m in matches if m.ruleId != "MORFOLOGIK_RULE_EN_US" ]
        metrics.summaryMatches = matches

        # -------- Bias metrics --------
        metrics.originalSentiment = self.sentimentPipeline(metrics.fullText, truncation=True)
        metrics.summSentiment = self.sentimentPipeline(spellSummary, truncation=True)

        # -------- Information theory metrics --------
        metrics.fullTextInfoTheory = self.infoTheory(fullTextNLP)
        metrics.chunksInfoTheory = [self.infoTheory(c) for c in chunksNLP]
        metrics.generalInfoTheory = self.infoTheory(generalNLP)
        metrics.series0InfoTheory = [self.infoTheory(s) for s in series0NLP]
        metrics.summInfoTheory = self.infoTheory(summNLP)
        metrics.fullSumm0InfoTheory = self.infoTheory(fullSumm0NLP)
        metrics.fullSummNInfoTheory = self.infoTheory(fullSummNNLP) if fullSummNNLP != None else None
        metrics.partSummInfoTheory = self.infoTheory(partSummNLP)

        return metrics

class TextMetrics:
    def __init__(self):
        self.start = False 
        self.fullText = ""

    def jsonPreprocess(self, json):
        if self.start:
            self.fullText += json['text'] + ".\n"
        else:
            if ("be it enacted" in json['text'].lower() or "be it therefore enacted" in json['text'].lower()):
                self.start = True

        if 'children' in json:
            for child in json['children']:
                self.jsonPreprocess(child)

    def __str__(self):
        ret = "\n---------------------------------------------\nText metrics\n---------------------------------------------\n"
        ret += "\tOriginal: \n\t" + str(self.fullTextStats) + "\n"
        # for key, value in self.fullTextStats.items():
        #     ret += "\t" + str(key) + ": " + "{:.6f}".format(value) + ",\n"
        # ret = ret[:-2]
        # ret += "\t}"
        ret += "General: \n\t" + str(self.generalStats) + "\n"
        ret += "Summary: \n\t" + str(self.summStats) + "\n"
        ret += "Partial Summ: \n\t" + str(self.partSummStats) + "\n"
        ret += "Full Summ: \n\t" + str(self.fullSumm0Stats) + "\n"
        ret += "Chunks: [\n"
        for i,c in enumerate(zip(self.chunksStats, self.series0Stats)):
            ret += "\tChunk " + str(i) + ":\n"
            ret += "\t"*2 + str(c[0])
            ret += "\n" + "\t"*2 + "-------- --------\n"
            ret += "\t"*2 + str(c[1]) + "\n"
        ret += "]\n"

        ret += "\n---------------------------------------------\nReadability metrics\n---------------------------------------------\n"
        ret += "Original: " + "\n"
        ret += self.readabilitystr(self.fullTextReadability) + "\n"
        ret += "General: \n" + self.readabilitystr(self.generalReadability) + "\n"
        ret += "Summary: \n" + self.readabilitystr(self.summReadability) + "\n"
        ret += "Partial Summ: \n" + self.readabilitystr(self.partSummReadability) + "\n"
        ret += "Full Summ: \n" + self.readabilitystr(self.fullSumm0Readability) + "\n"
        ret += "Chunk-Series: [\n"
        for i,c in enumerate(zip(self.chunksReadability, self.series0Readability)):
            ret += "\tChunk " + str(i) + ":" + "\n"
            ret += self.readabilitystr(c, depth=2)
        ret += "]\n"

        ret += "\n---------------------------------------------\nSummarisation metrics\n---------------------------------------------\n"
        ret += "Partial Summ: " + "\n"
        ret += "\tROUGE: " + str(self.fullPartSummRouge) + "\n"
        ret += "\tBLEU: " + str(self.fullPartSummBleu) + "\n"
        ret += "\tBERT: " + str(self.fullPartSummBert) + "\n"

        ret += "\nFull Summ: " + "\n"
        ret += "\tROUGE: " + str(self.full2Summ0Rouge) + "\n"
        ret += "\tBLEU: " + str(self.full2Summ0Bleu) + "\n"
        ret += "\tBERT: " + str(self.full2Summ0Bert) + "\n"
        
        ret += "\nChunk-Series: [\n"
        for i,c in enumerate(zip(self.chunkSimpRouge, self.chunkSimpBleu, self.chunkSimpBert)):
            ret += "\tChunk " + str(i) + ":" + "\n"
            ret += "\t\t" + "ROUGE: " + str(c[0]) + "\n"
            ret += "\t\t" + "BLEU: " + str(c[1]) + "\n"
            ret += "\t\t" + "BERT: " + str(c[2]) + "\n"
        ret += "]\n"

        ret += "\n---------------------------------------------\nOrthographic metrics\n---------------------------------------------\n"
        ret += "Original errors: " + str(len(self.originalMatches)) + "\n"
        ret += "Summary errors: " + str(len(self.summaryMatches)) + "\n"
        ret += "[\n"
        for match in self.summaryMatches:
            ret += "\t" + str(match.message) + ": \"" + str(match.context) + "\"\n"
        ret += "]\n" 

        ret += "\n---------------------------------------------\nBias metrics\n---------------------------------------------\n"
        ret += "Original sentiment: " + str(self.originalSentiment) + "\n"
        ret += "Summary sentiment: " + str(self.summSentiment) + "\n"

        ret += "\n---------------------------------------------\nInformation theory metrics\n---------------------------------------------\n"
        ret += "Original: \n" + str(self.fullTextInfoTheory) + "\n"
        ret += "General: \n" + str(self.generalInfoTheory) + "\n"
        ret += "Summary: \n" + str(self.summInfoTheory) + "\n"
        ret += "Partial Summ: \n" + str(self.partSummInfoTheory) + "\n"
        ret += "Full Summ: \n" + str(self.fullSumm0InfoTheory) + "\n"
        ret += "Chunk-Series: [\n"
        for i,c in enumerate(zip(self.chunksInfoTheory, self.series0InfoTheory)):
            ret += "\tChunk " + str(i) + ":" + "\n"
            ret += "\t"*2 + str(c[0])
            ret += "\n" + "\t"*2 + "-------- --------\n"
            ret += "\t"*2 + str(c[1]) + "\n"
        ret += "]\n"

        return ret

    def readabilitystr(self, r, depth=1):
        if (type(r) == tuple):
            ret = ""
            ret += "\t"*depth + "Flesch-Kincaid: " + "{:.6f}".format(r[0]['flesch_kincaid_grade'])[:9] + " | " + "{:.6f}".format(r[1]['flesch_kincaid_grade'])[:9] + "\n" 
            ret += "\t"*depth + "Flesch:         " + "{:.6f}".format(r[0]['flesch_reading_ease'])[:9] + " | " + "{:.6f}".format(r[1]['flesch_reading_ease'])[:9] + "\n"
            ret += "\t"*depth + "Gunning-Fog:    " + "{:.6f}".format(r[0]['gunning_fog'])[:9] + " | " + "{:.6f}".format(r[1]['gunning_fog'])[:9] + "\n"
            ret += "\t"*depth + "ARI:            " + "{:.6f}".format(r[0]['automated_readability_index'])[:9] + " | " + "{:.6f}".format(r[1]['automated_readability_index'])[:9] + "\n"
            ret += "\t"*depth + "Coleman-Liau:   " + "{:.6f}".format(r[0]['coleman_liau_index'])[:9] + " | " + "{:.6f}".format(r[1]['coleman_liau_index'])[:9] + "\n"
            ret += "\t"*depth + "SMOG:           " + "{:.6f}".format(r[0]['smog'])[:9] + " | " + "{:.6f}".format(r[1]['smog'])[:9] + "\n"
            ret += "\t"*depth + "Lix:            " + "{:.6f}".format(r[0]['lix'])[:9] + " | " + "{:.6f}".format(r[1]['lix'])[:9] + "\n"
            ret += "\t"*depth + "Rix:            " + "{:.6f}".format(r[0]['rix'])[:9] + " | " + "{:.6f}".format(r[1]['rix'])[:9] + "\n"
            return ret

        else:
            ret = ""
            ret += "\t"*depth + "Flesch-Kincaid: " + "{:.6f}".format(r['flesch_kincaid_grade'])[:9] + "\n" 
            ret += "\t"*depth + "Flesch:         " + "{:.6f}".format(r['flesch_reading_ease'])[:9] + "\n"
            ret += "\t"*depth + "Gunning-Fog:    " + "{:.6f}".format(r['gunning_fog'])[:9] + "\n"
            ret += "\t"*depth + "ARI:            " + "{:.6f}".format(r['automated_readability_index'])[:9] + "\n"
            ret += "\t"*depth + "Coleman-Liau:   " + "{:.6f}".format(r['coleman_liau_index'])[:9] + "\n"
            ret += "\t"*depth + "SMOG:           " + "{:.6f}".format(r['smog'])[:9] + "\n"
            ret += "\t"*depth + "Lix:            " + "{:.6f}".format(r['lix'])[:9] + "\n"
            ret += "\t"*depth + "Rix:            " + "{:.6f}".format(r['rix'])[:9] + "\n"
            return ret

    def to_dict(self):
        textMetrics = {
            'original': self.fullTextStats,
            'general': self.generalStats,
            'summ': self.summStats,
            'fullSumm0': self.fullSumm0Stats,
            'fullSummN': self.fullSummNStats,
            'partSumm': self.partSummStats
        }

        readabilityMetrics = {
            'original': self.fullTextReadability,
            'general': self.generalReadability,
            'summ': self.summReadability,
            'fullSumm0': self.fullSumm0Readability,
            'fullSummN': self.fullSummNReadability,
            'partSumm': self.partSummReadability
        }

        summarisationMetris = {
            'full_part': {
                'rouge': self.fullPartSummRouge,
                'bleu': self.fullPartSummBleu,
                'bert': self.fullPartSummBert
            },
            'full_summ0': {
                'rouge': self.full2Summ0Rouge,
                'bleu': self.full2Summ0Bleu,
                'bert': self.full2Summ0Bert
            },
            'full_summN': {
                'rouge': self.full2SummNRouge,
                'bleu': self.full2SummNBleu,
                'bert': self.full2SummNBert
            }
        }

        orthographicMetrics= {
            'original': len(self.originalMatches),
            'summary': len(self.summaryMatches)
        }

        biasMetrics = {
            'original': self.originalSentiment[0],
            'summary': self.summSentiment[0]
        }

        infoMetrics = {
            'original': self.fullTextInfoTheory,
            'general': self.generalInfoTheory,
            'summ': self.summInfoTheory,
            'fullSumm0': self.fullSumm0InfoTheory,
            'fullSummN': self.fullSummNInfoTheory,
            'partSumm': self.partSummInfoTheory
        }

        return {
            'text': textMetrics,
            'readability': readabilityMetrics,
            'summarisation': summarisationMetris,
            'orthographic': orthographicMetrics,
            'bias': biasMetrics,
            'info': infoMetrics
        }
