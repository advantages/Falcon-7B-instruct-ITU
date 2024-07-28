# Falcon-7B-instruct-ITU

lora weights can be downloaded at https://drive.google.com/drive/folders/1nhBwPR3DD25vXo8y-ZCX4ROkPA6caGju?usp=drive_link

For testing the results, one can directly run the following code:

```python
python test_falcon.py
```

Note that changing the "falcon-7b-instruct" path with your local path.

```python
model = AutoModelForCausalLM.from_pretrained(
    "/data/tppan/falcon-7b-instruct",
    return_dict=True,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained("/data/tppan/falcon-7b-instruct")
```

Each item in the provided csv file including three parts:

```python
{Question}

The following are the options towards the given question: {options}

The following are the prior knowledge you can refer to: {explanation}
```

Explanation parts are generated using claude-3-sonnet model with the following prompt, the detailed motivation of which will be provided in the workshop paper.

```python
You are an expert in telecommunications, your role is to provide rational explanation for the provided options. Your explanations should reveal the internal logic behind the questions and options, helping a layperson understand the reasoning without explicitly stating the answer.

Guidelines for explanation:

- Keep the explanations concise, typically within 100 words.
- Provide logical explanations that assist a layperson in identifying the correct answer among different options.
- Avoid explicit hints or directly stating the answer within the explanations.
- Focus on the key concepts, principles, or mechanisms relevant to the question and answer.

All the options for the question in json format: {{ Options }}

Question in json format: {{ Question }}
```
