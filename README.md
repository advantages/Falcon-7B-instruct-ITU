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
Explanation parts are generated using claude-3-sonnet model, which we will provide detailed introduction in the workshop paper.
