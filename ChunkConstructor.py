import math

class ChunkConstructor:
    def __init__(self, tokenizer, chunkMax=1024, clean=100):
        self.chunkMax = chunkMax
        self.clean = clean
        self.tokenizer = tokenizer
        self.startup(chunkMax)

    def startup(self, chunkMax):
        self.count=0
        self.chunkMax = chunkMax if chunkMax != None else self.chunkMax
        self.tempChunk = ""
        self.chunks = []
        self.start = False

        self.listTemp = ""

    def __call__(self, leg, type="Sectioned", chunkMax=None, newtokens=0):
        self.startup(chunkMax)
        if type=="Naive":
            self.naiveChunking(leg, -3)
            self.chunks.append({'length': len(self.tokenizer.tokenize(self.tempChunk)), 'text': self.tempChunk})
        elif type=="SectionedNaive":
            self.sectionedNaiveChunking(leg,-3)
            self.tempChunk += self.listTemp
            self.chunks.append({'length': len(self.tokenizer.tokenize(self.tempChunk)), 'text': self.tempChunk})
        elif type=="Sectioned":
            if leg['tokenSum'] < self.chunkMax:
                self.naiveChunking(leg, -3)
                self.chunks.append({'length': len(self.tokenizer.tokenize(self.tempChunk)), 'text': self.tempChunk})
            else:
                self.sectionChunking(leg)

        
        self.chunkSanitizer(self.clean)
        return self.chunks

    def naiveChunking(self,json, depth):
        if self.start:
            self.count = self.count + json['tokenCount']
            if (self.count < self.chunkMax):
                self.tempChunk = self.tempChunk + "\n" + "\t"*max(depth, 0) + json['text']
            else:
                self.count = json['tokenCount']
                self.chunks.append({'length': len(self.tokenizer.tokenize(self.tempChunk)), 'text': self.tempChunk})
                self.tempChunk = "\n" + "\t"*max(depth, 0) + json['text']
        else:
            if ("be it enacted" in json['text'].lower() or "be it therefore enacted" in json['text'].lower()):
                self.start = True

        if 'children' in json:
            for child in json['children']:
                self.naiveChunking(child, depth+1)

    def sectionedNaiveChunking(self, json, depth, addListChunk):
        if self.start:
            self.count = self.count + json['tokenCount']
            if (self.count < self.chunkMax):
                self.listTemp = self.listTemp + "\n" + "\t"*max(depth, 0) + json['text']
                if (addListChunk):
                    self.tempChunk = self.tempChunk + self.listTemp
                    self.listTemp = ""
            else: 
                self.count = json['tokenCount']
                self.chunks.append({'length': len(self.tokenizer.tokenize(self.tempChunk)), 'text': self.tempChunk})
                self.tempChunk = "\n" + "\t"*max(depth, 0) + json['text']
        else:
            if ("be it enacted" in json['text'].lower() or "be it therefore enacted" in json['text'].lower()):
                self.start = True

        if 'children' in json:
            for i, child in enumerate(json['children']):
                if i==len(json['children'])-1 and not ('children' in child):
                    self.sectionedNaiveChunking(child, depth+1, True)
                else:
                    self.sectionedNaiveChunking(child, depth+1, False)

    def sectionChunking(self, json):
        if self.start:
            if json['tokenSum'] < self.chunkMax:
                self.chunkBuild(json, 0)
                self.chunks.append({'length': json['tokenSum'], 'text': self.tempChunk})
                self.tempChunk = ""
            else:
                if 'children' in json:
                    for child in json['children']:
                        self.sectionChunking(child)
        else: 
            if ("be it enacted" in json['text'].lower() or "be it therefore enacted" in json['text'].lower()):
                self.start = True

            if 'children' in json:
                for child in json['children']:
                    self.sectionChunking(child)

    def chunkBuild(self, json, depth):
        self.tempChunk = self.tempChunk + "\n" + "\t"*max(depth, 0) + json['text']
        if 'children' in json:
            for child in json['children']:
                self.chunkBuild(child, depth+1)

    def chunkSanitizer(self, clean):
        i = 0
        while i<len(self.chunks):
            if self.chunks[i]['length'] < clean:
                del self.chunks[i]
            else:
                i+=1