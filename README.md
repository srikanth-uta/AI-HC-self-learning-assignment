
# Organ and associated diseases

This project is to create an LLM using Clinical BERT and RAG L7. Data used is from MIMIC III data of diagnoses codes and associated diseases. 

The pupose of this LLM is to give diseases associated with the organ.

Articles and links used to develop this are here

https://huggingface.co/emilyalsentzer/Bio_ClinicalBERT 




# Instructions to run the model

install dependencies:

```bash
pip install -r requirements.txt
```

Then run the model, 

``` bash
python3 models/slt_model.py
```

Once the model ran successfully, you can test the model by running quick_test

``` bash
python3 quick_test.py
```

This will prompt input, try to enter question like 

`What diseases can affect the liver?`

This should answer as 

`Cirrhosis of liver without mention of alcohol, Peritoneal adhesions (postoperative) (postinfection), Cholangitis, Other abnormal glucose, Inguinal hernia, without mention of obstruction or gangrene, bilateral (not specified as recurrent), Ulcerative (chronic) enterocolitis`

