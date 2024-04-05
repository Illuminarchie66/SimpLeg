from Pipeline.ChunkConstructor import ChunkConstructor
from Pipeline.LlamaChain import LlamaChain
from langchain.prompts import PromptTemplate
from Pipeline.LegislationBuilder import LegislationBuilder
from transformers import GenerationConfig

import time
import re
import copy
import math

class Generator:

    def __init__(self, baseModel="NousResearch/Llama-2-13b-chat-hf", verbose=False):
        
        self.baseModel = baseModel
        self.contextWindow = 4096

        # -------- Builders --------
        self.startTime = time.time()

        self.chainBuilder = LlamaChain(baseModel, verbose=False)
        self.legBuilder = LegislationBuilder(self.chainBuilder.tokenizer)
        self.chunkBuilder = ChunkConstructor(self.chainBuilder.tokenizer,chunkMax=self.contextWindow)

        print("Builders:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

        # -------- Chains --------
        self.initChains(verbose)

    def initChains(self, verbose=False):

        self.startTime = time.time()

        # -------- Single-gen chain --------
        self.genericConfig = GenerationConfig.from_pretrained(self.baseModel)
        self.genericConfig.max_new_tokens = 1024
        self.genericConfig.min_new_tokens = 0
        self.genericConfig.temperature = 0.00001
        self.genericConfig.top_p = 0.95
        self.genericConfig.do_sample = True
        self.genericConfig.repetition_penalty = 1.15
        self.genericConfig.num_beams = 1
        self.genericConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        self.initSingleChain()
        self.initChunkChain()
        self.initSeriesChain()
        self.initGeneralChain()
        self.initShortChain()
        self.initSummaryChain()
        
        print("Chains:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

    def initSingleChain(self):
        # -------- Single-gen chain --------
        singleGenConfig = GenerationConfig.from_pretrained(self.baseModel)
        singleGenConfig.max_new_tokens = 1500
        singleGenConfig.min_new_tokens = 200
        singleGenConfig.temperature = 0.00001
        singleGenConfig.top_p = 0.95
        singleGenConfig.do_sample = True
        singleGenConfig.repetition_penalty = 1.0
        singleGenConfig.num_beams = 1
        singleGenConfig.length_penalty = 1.0
        singleGenConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        template = """
Explain and simplify the text delimited by the triple backticks.
The text is a piece of British Legislation. Simplify what this bill does, and explain the effects of this bill, making it easier to understand.
Give your answer in only bullet points, explaining the key points of the text. Focus on what this legislation does, and who it impacts. Make it very easy to understand. 
```
TITLE: {title}
LONG TITLE: {longtitle}
CONTENTS: {body}
```
BULLET POINTS:
"""
        singleGenPrompt = PromptTemplate(template=template, input_variables=['title','longtitle','body'])
        self.singleGenChain = self.chainBuilder.loadChain(generationConfig=singleGenConfig, prompt=singleGenPrompt, name="SingleGen", verbose=True)

    def initChunkChain(self):
        # -------- Chunk-gen chain --------

        chunkGenConfig = GenerationConfig.from_pretrained(self.baseModel)
        chunkGenConfig.max_new_tokens = 4096
        chunkGenConfig.min_new_tokens = 0
        chunkGenConfig.temperature = 0.00001
        chunkGenConfig.top_p = 0.95
        chunkGenConfig.do_sample = True
        chunkGenConfig.repetition_penalty = 1.15
        chunkGenConfig.num_beams = 1 
        chunkGenConfig.length_penalty = 1.0
        chunkGenConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        template = """
The text delimited by the triple backticks are an extract of clauses from a piece of UK Legislation with the title: {title}
Simplify and explain what this extract of the legislation does.  
```{body}```
You MUST give your answer in this specific format: 
`* Simplified point 1:
* Simplified point 2:
...`
Use as many points as needed.
ANSWER:
"""
        chunkGenPrompt = PromptTemplate(template=template, input_variables=['body', 'title'])
        self.chunkGenChain = self.chainBuilder.loadChain(generationConfig=chunkGenConfig, prompt=chunkGenPrompt, name="ChunkGen", verbose=True) 

    def initSeriesChain(self):
        # -------- Series-gen chain --------

        template = """
You are an AI who takes in inputs of groups of lists of points, and you create a concise list of shorter, more simplified points. 
The simplified points are from a piece of UK legislation with the title: {title}
An example input will look like this: 
`Group 1
Simplified Point 1: <some text>
Simplified Point 2: <some text>

Group 2
Simplified Point 1: <some text>
...`
Simplify and explain what this extract of the legislation does. Focus on the points which are likely to be the most impactful.
You MUST give your answer in this specific format: 
`* Simplified point 1:
* Simplified point 2:
...`
Use as many points as needed.
INPUT: {body}
ANSWER:
"""
        seriesConfig = GenerationConfig.from_pretrained(self.baseModel)
        seriesConfig.max_new_tokens = 4096
        seriesConfig.min_new_tokens = 0
        seriesConfig.temperature = 0.000001
        seriesConfig.top_p = 0.95
        seriesConfig.do_sample = True
        seriesConfig.repetition_penalty = 1.15
        seriesConfig.num_beams = 1
        seriesConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        seriesGenPrompt = PromptTemplate(template=template, input_variables=['title','body'])
        self.seriesGenChain = self.chainBuilder.loadChain(generationConfig=seriesConfig, prompt=seriesGenPrompt, name="SeriesGen", verbose=True)

    def initGeneralChain(self):
        # -------- General chain --------
        template = """
Given is the title and long title of a piece of British Legislation. 
Explain what this legislation does in very simple terms. Make it very easy to understand what it is doing. Ensure that your answer is appropriate and respectful.
TITLE: {title}
LONG TITLE: {longtitle}
SIMPLE EXPLAINATION:
"""

        generalConfig = GenerationConfig.from_pretrained(self.baseModel)
        generalConfig.max_new_tokens = 1024
        generalConfig.min_new_tokens = 0
        generalConfig.temperature = 0.00001
        generalConfig.top_p = 0.95
        generalConfig.do_sample = True
        generalConfig.repetition_penalty = 1.15
        generalConfig.num_beams = 1
        generalConfig.length_penalty = -0.1
        generalConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        generalGenPrompt = PromptTemplate(template=template, input_variables=['title','longtitle'])
        self.generalGenChain = self.chainBuilder.loadChain(generationConfig=generalConfig, prompt=generalGenPrompt, name="GeneralGen", verbose=True)

    def initShortChain(self):
        # -------- Short title chain --------
        shortTitleConfig = GenerationConfig.from_pretrained(self.baseModel)
        shortTitleConfig.max_new_tokens = 128
        shortTitleConfig.min_new_tokens = 0
        shortTitleConfig.temperature = 0.00001
        shortTitleConfig.top_p = 1.0
        shortTitleConfig.do_sample = True
        shortTitleConfig.repetition_penalty = 1.15
        shortTitleConfig.length_penalty = -0.4
        shortTitleConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        template1 = """
You are an AI who takes inputs of small blocks of text, and gives it a short descriptive title.
Your answer is to be short, and should try to best describe what the overall text means.
An example input will look like this: 
`Group 1
Simplified Point 1: <some text>
Simplified Point 2: <some text>

Group 2
Simplified Point 1: <some text>
...`
You MUST make your answer like a title.
INPUT: ```{body}```
ANSWER:
"""

        template2 = """
You are an AI who takes inputs of small blocks of text, and gives it a short descriptive title.
Your answer is to be short, and should try to best describe what the overall text means. The text you will be given is a summary of legislative text.
You MUST make your answer like a title.
INPUT:
"""
        shortTitlePrompt = PromptTemplate(template=template1, input_variables=['body'])
        self.shortTitleChain = self.chainBuilder.loadChain(generationConfig=shortTitleConfig, prompt=shortTitlePrompt, name="ShortTitleGen", verbose=True)

    def initSummaryChain(self):
        template = """
You are an AI which takes an input of a list of simplified points from a piece of UK Legislation, and makes it easier for readers to understand what the legislation does. 
You are to explain what this piece of UK Legislation does in simple and easy to understand terms. Make your answer easy to read, using simpler words and phrases, while trying not to lose as much information as possible.
Your output should be descriptive but easy for anybody to understand. You are encouraged to use simpler words and terms where appropriate. 
Focus your summary on the impacts of the legislation, and how it affects people. Ensure your summary is appropriate and respectful. 
TITLE: {title}
LONG TITLE: {longtitle}
INPUT: {body}
OUTPUT:
"""
        summConfig = GenerationConfig.from_pretrained(self.baseModel)
        summConfig.max_new_tokens = 4096
        summConfig.min_new_tokens = 0
        summConfig.temperature = 0.00001
        summConfig.top_p = 0.95
        summConfig.do_sample = True
        summConfig.repetition_penalty = 1.15
        summConfig.num_beams = 1
        summConfig.length_penalty = 1.5
        summConfig.eos_token_id = self.chainBuilder.tokenizer.eos_token_id

        summGenPrompt = PromptTemplate(template=template, input_variables=['title','longtitle', 'body'])
        self.summaryChain = self.chainBuilder.loadChain(generationConfig=summConfig, prompt=summGenPrompt, name="SummGen", verbose=True)

    def initSimplifyChain(self):
        template = """
You are an AI which takes in an input text, and performs small changes on it to make it more readable. 
This involves taking complex ideas and making them more simple; replacing complex words with easier ones; and explaining any difficult ideas present in the text.
INPUT: {body}
OUTPUT:
"""
        simplifyGenPrompt = PromptTemplate(template=template, input_variables=["body"])
        self.simplifyChain = self.chainBuilder.loadChain(generationConfig=self.genericConfig, prompt=simplifyGenPrompt, name="SimpGen", verbose=True)

    def generate(self, legislation,verbose=False, simplifiedChunksFlag=False):
        def sanitize(result):
            result['text'] = re.sub(r'\n+', '\n', result['text'])

        buffer = 256 + len(self.chainBuilder.tokenizer.tokenize(legislation.title))
        clean = 128

        chunkSummaries = []
        legislation.chunks = self.chunkBuilder(legislation.json, chunkMax=self.contextWindow-buffer)
        legislation.writeChunks()

        self.startTime = time.time()
        general = self.generalGenChain.invoke({'title': legislation.title, 'longtitle': legislation.longtitle})
        print("General generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

        if len(legislation.chunks) == 1:

            self.startTime = time.time()
            singleInput = {'title': legislation.title, 'longtitle': legislation.longtitle, 'body': legislation.chunks[0]['text']}
            single = self.singleGenChain.invoke(singleInput)
            sanitize(single)
            print("Single generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            self.startTime = time.time()
            seriesInput = {'body': legislation.chunks[0]['text'], 'title':legislation.title}
            series = self.seriesGenChain.invoke(seriesInput)
            sanitize(series)
            print("Series generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None
            
            self.startTime = time.time()
            shortInput = {'body': series['text']}
            short = self.shortTitleChain.invoke(shortInput)
            sanitize(short)
            retSeries = {'length': len(self.chainBuilder.tokenizer.tokenize(short['text'] + series['text'])), 'short': short['text'], 'text': series['text']}
            print("Short generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            self.startTime = time.time()
            summaryInput = {'title': legislation.title, 'longtitle': legislation.longtitle, 'body':series['text']}
            summary = self.summaryChain.invoke(summaryInput)
            sanitize(summary)
            print("Summary generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            if simplifiedChunksFlag:
                simplifiedChunks = []
                for chunk in [retSeries]:
                    s = self.simplifyChain.invoke({'body':chunk['short'] + "\n" + chunk['text']})
                    simplifiedChunks.append(s['text'])
            else:
                simplifiedChunks = None

            return {'general': general['text'], 'series': [[retSeries]], 'simp_series': simplifiedChunks, 'summary': summary['text'], 'single': single['text']}
        else:
            simps = []
            newInput = ""
            index = 0

            self.startTime = time.time()
            for chunk in legislation.chunks:
                simpInput = {'body': chunk['text'], 'title': legislation.title}
                simp = self.chunkGenChain.invoke(simpInput)
                sanitize(simp)

                shortInput = {'body': simp['text']}
                short = self.shortTitleChain.invoke(shortInput)
                sanitize(short)

                ret = short['text'] + "\n" + simp['text'] 
                simps.append({'length': len(self.chainBuilder.tokenizer.tokenize(short['text'] + simp['text'])), 'short': short['text'], 'text': simp['text']})
                newInput += ret + "\n"
            
            chunkSummaries.append(simps[:])
            newtokens = len(self.chainBuilder.tokenizer.tokenize(newInput))
            print("Simp " + str(index) + " generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            if (newtokens > self.contextWindow-buffer):
                while (newtokens > self.contextWindow-buffer):
                    index += 1
                    self.startTime = time.time()

                    needed = math.ceil(newtokens/(self.contextWindow-buffer))
                    chunks = []
                    newChunk = ""
                    tokens = 0
                    t = 1

                    for s in simps:
                        newChunk += s['text'] + "\n"
                        tokens += 1 + s['length']
                        if tokens/newtokens >= t/needed:
                            chunks.append(newChunk)
                            newChunk = ""
                            t += 1
                    chunks.append(newChunk)

                    i = 0
                    while i<len(chunks):
                        if len(self.chainBuilder.tokenizer.tokenize(chunks[i])) < clean:
                            del chunks[i]
                        else:
                            i+=1

                    simps.clear()
                    newInput = ""
                    for c in chunks:
                        seriesInput = {'body': c, 'title': legislation.title}
                        simp = self.seriesGenChain.invoke(seriesInput)
                        sanitize(simp)

                        shortInput = {'body': simp['text']}
                        short = self.shortTitleChain.invoke(shortInput)
                        sanitize(short)

                        simps.append({'length': len(self.chainBuilder.tokenizer.tokenize(simp['text'])), 'short': short['text'], 'text': simp['text']})
                        newInput += short['text'] + "\n" + simp['text'] + "\n"

                    chunkSummaries.append(simps[:])
                    newtokens = len(self.chainBuilder.tokenizer.tokenize(newInput))
                    print("Simp " + str(index) + " generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None
            
            self.startTime = time.time()
            seriesInput = {'body': newInput, 'title': legislation.title}
            series = self.seriesGenChain.invoke(seriesInput)
            sanitize(series)
            print("Series generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            self.startTime = time.time()
            shortInput = {'body': series['text']}
            short = self.shortTitleChain.invoke(shortInput)
            sanitize(short)
            print("Short generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            self.startTime = time.time()
            finalSeries = [{'length': len(self.chainBuilder.tokenizer.tokenize(series['text'])), 'short': short['text'], 'text': series['text']}]
            chunkSummaries.append(finalSeries)
            print("Simp " + str(index+1) + " generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            self.startTime = time.time()
            summaryInput = {'title': legislation.title, 'longtitle': legislation.longtitle, 'body':newInput}
            summary = self.summaryChain.invoke(summaryInput)
            sanitize(summary)
            print("Summary generate:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

            if simplifiedChunksFlag:
                simplifiedChunks = []
                for chunk in chunkSummaries[0]:
                    s = self.simplifyChain.invoke({'body':chunk['short'] + "\n" + chunk['text']})
                    simplifiedChunks.append(s['text'])
            else:
                simplifiedChunks = None

            return {'general': general['text'],  'series': chunkSummaries, 'simp_series': simplifiedChunks, 'summary': summary['text']}



