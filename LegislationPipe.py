import requests
import html2text
import re
from lxml import html
from bs4 import BeautifulSoup
import time

class TreeNode:
    def __init__(self, value, text, p, textType, tokenSum, prefix=""):
        self.order = value
        self.text = text
        self.count = 0
        self.tokenSum = tokenSum
        self.prefix = prefix
        self.parent = p
        self.type = textType
        self.children = []

    def setCount(self, count):
        self.count = count

    def editText(self, newText):
        self.text = newText

class LegislationPipe:
    @staticmethod
    def getLegislationHTML(year, id):
        callString = "https://www.legislation.gov.uk/ukpga/{year}/{id}/enacted?view=plain".format(year=year, id=id)
        response = requests.get(callString)
        if response.ok:
            soup = BeautifulSoup(str(response.text), 'lxml')
            return soup
        else:
            return None
        
    @staticmethod
    def getLegislationHTMLTitle(soup):
        return soup.find(attrs={'class':'LegTitle'}).text
    
    @staticmethod
    def getLegislationHTMLLongTitle(soup):
        return soup.find(attrs={'class':'LegLongTitle'}).text
    
    @staticmethod
    def getLegislationHTMLDate(soup):
        return soup.find(attrs={'class':'LegDateOfEnactment'}).text

    @staticmethod 
    def getLegislationHTMLBody(soup):
        return soup.find(attrs={'id':'viewLegSnippet'})
        
    @staticmethod
    def getLegislatonString(content):
        if content != None:
            return html2text.html2text(str(content))
    
    @staticmethod
    def getLegislationTree(pureString, tokenizer):
        result = re.split(r'\n\n', pureString)
        resultm = [string.replace('\n', '') for string in result]

        metaDepth = 0
        currentDepth = 0
        localDepth = 0

        sameLine = False
        indent = False

        tokenSum = 0

        root = TreeNode(-1, "", None, "", 0, "")
        trueRoot = root

        for line in resultm:
            words = line.split(" ")
            type = ""
            if (re.match(r'(#+)(\s)*Schedule(s)*',line, re.IGNORECASE)):
                break

            if (sameLine) :
                root.editText(root.text+line)
                tokenSum = tokenSum + len(tokenizer.tokenize(line))
                sameLine = False
            else:
                if (re.match(r'#+' ,words[0])): #title
                    localDepth = words[0].count('#')
                    metaDepth = localDepth
                    currentDepth = localDepth

                    type = "Heading"
                    if (re.match(r'(\d)+(\D)+', words[1])):
                        pre, start = re.match(r'(\d+)(\D+)', words[1]).groups()
                        prefix = words[0] + " " + pre
                        text = start + ' ' + ' '.join(words[2:])
                    else:
                        prefix = words[0]
                        text = ' '.join(words[1:])

                    indent = True

                elif (re.match(r'(“)*(\d)*\((\d)+\)', words[0])): #sub bit
                    localDepth = metaDepth+1
                    currentDepth = localDepth
                    type = "Digit point"

                    split = int((re.search(r'(“)*(\d)*\((\d)+\)', words[0])).end())
                    
                    prefix = words[0][:split]
                    text = words[0][split:] + " " + " ".join(words[1:])
                    
                elif (re.match(r'(“)*(\d)*\(([a-z])+\)', words[0])):
                    x = str(re.search(r'\(([a-z])+\)', words[0]))
                    x = (x[1:-1])
                    if re.match(r'[i|ii|iii|iv|v|vi|vii|viii|ix|x]', x):
                        localDepth = metaDepth+3
                        type = "Roman point"
                    else:
                        localDepth = metaDepth+2
                        type = "Alpha point"

                    currentDepth = localDepth
                    if (re.match(r'(“)*(\d)*\(([a-z])+\)\s*$', words[0])):
                        sameLine = True
                        prefix = words[0]
                        text = ""
                    else:
                        split = int((re.search(r'(“)*(\d)*\(([a-z])+\)', words[0])).end())
                        prefix = words[0][:split]
                        text = words[0][split:] + " " + " ".join(words[1:])

                elif (re.match(r'(\s)*\*(\s)*.*', line)):
                    localDepth = currentDepth+1   
                    type = "Bullet point" 

                    prefix = "*"
                    index = -1
                    for i,c in enumerate(line):
                        if c != " " or c != "*":
                            index = i
                            break

                    text = line[index:]
                
                else:
                    if indent:
                        localDepth = currentDepth+1
                        indent = False
                    else:
                        localDepth = currentDepth
                        
                    type = "Text"

                    prefix = ""
                    text = line

                tokenSum = tokenSum + len(tokenizer.tokenize(text))
                if (localDepth > root.order):
                    # Text is a child of the previous value
                    tmp = TreeNode(localDepth, text, root, type, tokenSum, prefix)
                    root.children = root.children + [tmp]
                    root = tmp
                elif (localDepth == root.order):
                    # Text is a uncle of previous value
                    tmp = TreeNode(localDepth, text, root.parent, type, tokenSum, prefix)
                    root.parent.children = root.parent.children + [tmp]
                    root = tmp 
                else:
                    # Text is some distant ancestor of previous value
                    while (root.order != localDepth-1):
                        root = root.parent
                    tmp = TreeNode(localDepth, text, root, type, tokenSum, prefix)
                    root.children = root.children + [tmp]
                    root = tmp

        return trueRoot

    @staticmethod
    def getLegislationJSON(node):
        if not node.children:
            return {"type": node.type, "prefix": node.prefix, "text": node.text, "tokenCum": node.tokenSum}
        else:
            return {"type": node.type, "prefix": node.prefix, "text": node.text,  "tokenCum": node.tokenSum, "children": [LegislationPipe.getLegislationJSON(child) for child in node.children]}
    
    @staticmethod
    def getChildren(json):
        if 'children' in json:
            total = len(json['children'])
            for child in json['children']:
                total = total + LegislationPipe.getChildren(child)
            return total
        else:
            return 1
        
    @staticmethod
    def setJSONLengths(json, tokenizer, depth = -3):
        if 'children' in json:
            wordSum = len(json['text'].split())
            tokenCount = len(tokenizer.tokenize(json['text']))
            tokenSum = tokenCount
            for child in json['children']:
                wordSum = wordSum + LegislationPipe.setJSONLengths(child, tokenizer, depth+1)[0]
                tokenSum = tokenSum + LegislationPipe.setJSONLengths(child, tokenizer, depth+1)[1]
            json['wordSum'] = wordSum
            json['tokenSum'] = tokenSum + 1+ max(depth,0)
            json['tokenCount'] = tokenCount + max(depth, 0)
            return [wordSum, tokenSum+1+max(depth,0)]
        else:
            json['wordSum'] = len(json['text'].split())
            tmp = len(tokenizer.tokenize(json['text']))
            json['tokenSum'] = tmp + 1 + max(depth,0) 
            json['tokenCount'] = tmp + 1 + depth
            return [len(json['text'].split()), tmp+1+max(depth,0)]     