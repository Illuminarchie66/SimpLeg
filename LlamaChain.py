from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    pipeline,
    GenerationConfig
)

baseModel = "NousResearch/Llama-2-13b-chat-hf"


# from langchain import ( 
#     PromptTemplate, 
#     HuggingFacePipeline, 
#     LLMChain 
# )

from langchain.prompts import PromptTemplate
from langchain_community.llms.huggingface_pipeline import HuggingFacePipeline
from langchain.chains import LLMChain

import torch
import time

template = """
Write a summary of the following text delimited by triple backticks.
Return your response which covers the key points of the text.
```{text}```
SUMMARY:
"""
basePrompt = PromptTemplate(template=template, input_variables=["text"])

class LlamaChain:
    def __init__(self, baseModel, verbose=False):
        self.startTime = time.time()

        # -------- Quantization config --------
        compute_dtype = getattr(torch, "float16")
        quant_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=compute_dtype,
            bnb_4bit_use_double_quant=False,
        )

        # -------- Model load --------
        self.model = AutoModelForCausalLM.from_pretrained(
            baseModel,
            quantization_config=quant_config,
            device_map={"": 0}
        )
        self.model.config.use_cache = False
        self.model.config.pretraining_tp = 1
        print("Model:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None
        self.startTime = time.time()

        # -------- Tokenizer load --------
        self.tokenizer = AutoTokenizer.from_pretrained(baseModel, trust_remote_code=True)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        print("Tokenizer:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None
        self.startTime = time.time()

    def loadChain(self, generationConfig, prompt=basePrompt, name="Langchain", verbose=False):
        self.startTime = time.time()
        pipe = pipeline(
            "text-generation",
            model=self.model,
            tokenizer=self.tokenizer,
            generation_config=generationConfig
        )
        print("Pipeline:\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None

        llm = HuggingFacePipeline(pipeline = pipe, batch_size=4, model_kwargs = {'temperature':0})
        chain = LLMChain(prompt=prompt, llm=llm)
        print(name + ":\n--- %s seconds ---" % (time.time() - self.startTime)) if verbose else None
        return chain
        

