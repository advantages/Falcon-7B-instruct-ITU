import os
# from pprint import pprint
# import json
import sys
import bitsandbytes as bnb
import pandas as pd
import torch
import csv
from peft import *
import torch.nn as nn
import transformers
from datasets import load_dataset
from huggingface_hub import notebook_login
from peft import (
    LoraConfig,
    PeftConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
)
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)

DEVICE = "cuda:0"


with open('./falconinstruct_newft.csv',encoding='utf-8') as csvfile:
    reader=csv.reader(csvfile)
    reader=list(reader)




with open('./SampleSubmission.csv',encoding='utf-8') as csvfileori:
    readerori=csv.reader(csvfileori)
    readerori=list(readerori)

data = {
    'question':[readerori[0][0]],
    'answer':[readerori[0][1]]
}
df = pd.DataFrame(data)
df.to_csv('./answer.csv', mode='a', index=False, header=False)





prompt = f"""
<human>: Who appoints the Chief Justice of India?
<assistant>:
""".strip()


bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    load_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)


# peft_model_id = "experiments_myown/checkpoint-1500"      # 0.803278688
peft_model_id = "./checkpoint-4000"

config = PeftConfig.from_pretrained(peft_model_id)
model = AutoModelForCausalLM.from_pretrained(
    "/data/tppan/falcon-7b-instruct",
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("/data/tppan/falcon-7b-instruct")
tokenizer.pad_token = tokenizer.eos_token


model = PeftModel.from_pretrained(model, peft_model_id)

generation_config = model.generation_config
generation_config.max_new_tokens = 200
generation_config.do_sample = True
generation_config.temperature = 0.01
# generation_config.top_p = 0.7
generation_config.num_return_sequences = 1
generation_config.pad_token_id = tokenizer.eos_token_id
generation_config.eod_token_id = tokenizer.eos_token_id

def generate_response(question: str) -> str:
    prompt = f"""
    <human>: {question}
    <assistant>:
    """.strip()
    encoding = tokenizer(prompt, return_tensors="pt").to(DEVICE)
    with torch.inference_mode():
        outputs = model.generate(
            input_ids=encoding.input_ids,
            attention_mask=encoding.attention_mask,
            generation_config=generation_config,
        )
    response = tokenizer.decode(outputs[0],skip_special_tokens=True)

    assistant_start = '<assistant>:'
    response_start = response.find(assistant_start)
    return response[response_start + len(assistant_start):].strip()




for i in range(len(reader)):
    try:
        text_input = reader[i][0]
        output=generate_response(text_input)

        with open("output.md",'a+') as file_o:
            file_o.write("question {}".format(i))
            file_o.write("\n")
            file_o.write("\n")
            file_o.write(output)
            file_o.write("\n***************************************\n\n")


        output=output.split("I choose option ")[1].strip()
        option_num=output[:1]


        data = {
            'question': [readerori[i+1][0]],
            'answer': [option_num]
        }

        df = pd.DataFrame(data)
        df.to_csv('./answer.csv', mode='a', index=False, header=False)
        print("answer:    {}".format(option_num),end="   ")




    except Exception as e:
        print(f'error file:{e.__traceback__.tb_frame.f_globals["__file__"]}')
        print(f"error line:{e.__traceback__.tb_lineno}")

        data = {
            'question': [readerori[i+1][0]],
            'answer': ['Null'],
        }


        df = pd.DataFrame(data)
        df.to_csv('./answer.csv', mode='a', index=False, header=False)
        print(i,end="   ")
        print("answer:    NULL",end="   ")
