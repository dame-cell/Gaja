# Gaja

We release Gaja, a Hindi/Hinglish chat model instruction finetuned on SarvamAI's OpenHathi model.

<p align="center" width="100%">
    <img src="asset\gajendra.jpg" alt="Gajendra is a Hindi/Hinglish instruction-tuned model based on different instruct datasets." style="width: 50%; min-width: 600px; display: block; margin: auto;">
</p>

<p align="center" width="100%" style="color: gray;">
     The word "gaja" (Sanskrit: गज) is a term in Sanskrit, the classical language of ancient India, and it translates to "elephant" in English
</p>


This repository contains the code for  "Gaja", a project focused on Instruct-Fine-tuning SarvamAI's OpenHathi model. which employs the LoRA methodology for efficient fine tuning. 

# Important Stuff to know 

* The total rows of the instruct dataset contains about 50k rows 
* The entire finetune process was done in a free goggle colab using the t4 gpu 
* The entire dataset was jsut a combination of different dataset from hugging face hub
* The total amount of dataset used here is - 5 


# Dataset Information

| dataset name | Author | Link |
|----------|----------|----------|
| hindi_instruct_v | smangrul |[hindi-instruct-v](https://huggingface.co/datasets/smangrul/hindi_instruct_v1)|
| alpaca-gpt4-hindi-hinglish | NebulaByte | [alpaca-gpt4-hindi-hinglish](https://huggingface.co/datasets/NebulaByte/alpaca-gpt4-hindi-hinglish) |
| indic-instruct-data-v0.1 | ai4bharat |[indic-instruct-data-v0.1](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1) |
| databricks-dolly-15k-Hindi| aaditya|[databricks-dolly-15k-Hindi](https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hindi)|
| databricks-dolly-15k-Hinglish-Codemix | aaditya|[databricks-dolly-15k-Hinglish-Codemix](https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hinglish-Codemix)|

# Example output 
| Instruction | Input | Response |
|-------------|-------|----------|
| why is machine learning so hard?mujhe batado please?|  |Machine learning is a complex field that involves the development and application of algorithms that enable machines to learn from data. While it is a fascinating and exciting field, it can be challenging to master due to its complexity and the sheer volume of data involved.  Machine learning is challenging because it involves a combination of various disciplines, including statistics, mathematics, computer science, and engineering. It also requires a deep understanding of the underlying principles of machine learning, including probability theory, decision trees, and artificial neural networks. Moreover, machine learning involves the processing and analysis of vast amounts of data, which can be overwhelming for even the most skilled professionals. It requires a significant investment of time and resources to develop and test machine learning models, and it can be difficult to identify the most effective algorithms and techniques. Overall, machine learning is a challenging field that requires a deep understanding of mathematics, statistics, and computer science, as well as the ability to process and analyze large amounts of data. It is a field that requires a significant investment of time and resources, and it can be difficult to identify the most effective algorithms and techniques. Despite these challenges, machine learning is an exciting and rapidly evolving field that has the potential to transform industries and improve our lives.|
| क्वांटम भौतिकी में हीजनवारण क्या है, और यह क्यों महत्वपूर्ण है? (What is superposition in quantum physics, and why is it important?)| | क्वांटम भौतिकी में हीजनवारण एक ऐसा घटना है जिसमें एक ही कण दो अलग-अलग स्थानों पर मौजूद हो सकता है। यह एक मौलिक भौतिकी का नियम है जो क्वांटम भौतिकी के क्वांटम यांत्रिकी के सिद्धांतों के माध्यम से समझाया गया है। हीजनवारण के पीछे का सिद्धांत यह है कि क्वांटम कणों में एक क्वांटम अवस्था होती है, जो एक विशेष गुण है जो उन्हें एक विशेष तरीके से कार्य करने की अनुमति देता है। इस गुण के कारण, हीजनवारण संभव है। हीजनवारण की अवधारणा क्वांटम भौतिकी के कई मौलिक सिद्धांतों के लिए महत्वपूर्ण है, जिसमें क्वांटम टनलिंग, क्वांटम बिहेवियर और क्वांटम जानकारी शामिल हैं। हीजनवारण के पीछे के सिद्धांतों को समझने से क्वांटम कंप्यूटिंग के क्षेत्र में प्रगति हो सकती है और क्वांटम यांत्रिकी के सिद्धांतों को बेहतर समझने में मदद मिल सकती है।|
| what does the word sophicticated mean?, answer in english then translate the answer  to hindi| | Sophisticated is a word that describes something that is very sophisticated or refined. It is often used to describe a person who is very cultured and has a high level of education and refined taste. It can also be used to describe a product or service that is of high quality and has been carefully crafted.  Translation: सौम्वरण एक ऐसा शब्द है जो किसी ऐसी चीज़ का वर्णन करता है जो बहुत ही सौम्वरण या परिष्कृत होती है। इसका उपयोग अक्सर किसी ऐसे व्यक्ति का वर्णन करने के लिए किया जाता है जो बहुत ही संस्कृत है और जिसकी उच्च स्तर की शिक्षा और परिष्कृत स्वाद है। इसका उपयोग किसी ऐसे उत्पाद या सेवा का वर्णन करने के लिए भी किया जा सकता है जो उच्च गुणवत्ता का हो और जिसे सावधानीपूर्वक तैयार किया गया हो।</s> |

# Prompting Format for Instruction Models
The prompt template for this Model follows the alpaca format 
1) The prompt template for only instruction
```python
{system prompt}
### Instruction:
{}

### Response:
{}
```
2) The prompt template with Input
   
```python
{system prompt}
### Instruction:
{}

### Input:
{}

### Response:
{}
```

# Local Inference 
You can try the model out in Ollama in your locally 
Steps to follow:
-  Download any of the gguf model you want from [Gaja.gguf](https://huggingface.co/damerajee/gaja-gguf)
-  install ollama either for Mac or Linux
-  Go to the folder Local inference and copy the Contents inside the Modelfile
-  create a txt file rename it Modelfile paste the Contents from inside the Modelfile to the text file you just created
-  Open your command prompt based on your device and os (make sure your in the same directory as where you saved your txt file and your gguf format model)
-  Type ollama create choose-a-model-name -f <location of the file e.g. ./Modelfile>'
-  ollama run choose-a-model-name
  For more Better Instruction please follow this docs -> [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

# Usage Note
It's important to note that the models have not undergone detoxification. Therefore, while they possess impressive linguistic capabilities, there is a possibility for them to generate content that could be deemed harmful or offensive. We urge users to exercise discretion and supervise the model's outputs closely, especially in public or sensitive applications.
