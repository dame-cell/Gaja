# Gaja

We release Gaja , a Hindi/Hinglish chat model instruction finetuned on SarvamAI's OpenHathi model.
And Dairava which is a merge of Gaja and Ai4bharat instruct model "Airavata"


<div text-align: center;">
  <img src="asset\Dariava.jpg" alt="Gajendra is a Hindi/Hinglish instruction-tuned model based on different instruct datasets." style="width: 45%; min-width: 300px;">
</div>


This repository contains the code for  "Gaja", a project focused on Instruct-Fine-tuning SarvamAI's OpenHathi model. which employs the LoRA methodology for efficient fine tuning. 

# Contents 
1) [Important Stuff to know](#important-stuff-to-know)
2) [Gaja Model-Based Variant](#gaja-model-based-variant)
3) [Dataset Information](#dataset-information)
4) [Gaja-Example output](#gaja-example-output)
5) [Dairava-Example output](#dairava-example-output)
6) [Gaja-Prompt Format](#gaja-prompt-format)
7) [Dairava-prompt Format](#dairava-prompt-format)
8) [Local Inference](#local-inference)
9) [Usage Note](#usage-note)


If you appreciate this work and found it helpful, consider giving it a star ⭐️ on GitHub. Your support motivates me to continue improving and adding new features. Thank you for your encouragement!

# Important Stuff to know 

* The total rows of the instruct dataset contains about 50k rows 
* The entire finetune process was done in a free goggle colab using the t4 gpu 
* The entire dataset was jsut a combination of different dataset from hugging face hub
 

# Gaja Model-Based Variant

| Model              | Type   | Data       | Base Model | # Params | Download Links |
|--------------------|--------|------------|------------|----------|----------------|
| Gaja               | Variant| 50k-instruct| Open-Hathi  | 7B     | [HF-HUB](https://huggingface.co/damerajee/Gaja)|
| Dairava           | merge |            | Open-Hathi | 7B     |[HF-HUB](https://huggingface.co/damerajee/Dairava)|


# Dataset Information

| dataset name | Author | Link |
|----------|----------|----------|
| hindi_instruct_v | smangrul |[hindi-instruct-v](https://huggingface.co/datasets/smangrul/hindi_instruct_v1)|
| alpaca-gpt4-hindi-hinglish | NebulaByte | [alpaca-gpt4-hindi-hinglish](https://huggingface.co/datasets/NebulaByte/alpaca-gpt4-hindi-hinglish) |
| indic-instruct-data-v0.1 | ai4bharat |[indic-instruct-data-v0.1](https://huggingface.co/datasets/ai4bharat/indic-instruct-data-v0.1) |
| databricks-dolly-15k-Hindi| aaditya|[databricks-dolly-15k-Hindi](https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hindi)|
| databricks-dolly-15k-Hinglish-Codemix | aaditya|[databricks-dolly-15k-Hinglish-Codemix](https://huggingface.co/datasets/aaditya/databricks-dolly-15k-Hinglish-Codemix)|

# Gaja-Example output 
| Instruction | Input | Response |
|-------------|-------|----------|
| why is machine learning so hard?mujhe batado please?| (No specific input provided)  |Machine learning is a complex field that involves the development and application of algorithms that enable machines to learn from data. While it is a fascinating and exciting field, it can be challenging to master due to its complexity and the sheer volume of data involved.  Machine learning is challenging because it involves a combination of various disciplines, including statistics, mathematics, computer science, and engineering. It also requires a deep understanding of the underlying principles of machine learning, including probability theory, decision trees, and artificial neural networks. Moreover, machine learning involves the processing and analysis of vast amounts of data, which can be overwhelming for even the most skilled professionals. It requires a significant investment of time and resources to develop and test machine learning models, and it can be difficult to identify the most effective algorithms and techniques. Overall, machine learning is a challenging field that requires a deep understanding of mathematics, statistics, and computer science, as well as the ability to process and analyze large amounts of data. It is a field that requires a significant investment of time and resources, and it can be difficult to identify the most effective algorithms and techniques. Despite these challenges, machine learning is an exciting and rapidly evolving field that has the potential to transform industries and improve our lives.|
| क्वांटम भौतिकी में हीजनवारण क्या है, और यह क्यों महत्वपूर्ण है? (What is superposition in quantum physics, and why is it important?)|(No specific input provided) | क्वांटम भौतिकी में हीजनवारण एक ऐसा घटना है जिसमें एक ही कण दो अलग-अलग स्थानों पर मौजूद हो सकता है। यह एक मौलिक भौतिकी का नियम है जो क्वांटम भौतिकी के क्वांटम यांत्रिकी के सिद्धांतों के माध्यम से समझाया गया है। हीजनवारण के पीछे का सिद्धांत यह है कि क्वांटम कणों में एक क्वांटम अवस्था होती है, जो एक विशेष गुण है जो उन्हें एक विशेष तरीके से कार्य करने की अनुमति देता है। इस गुण के कारण, हीजनवारण संभव है। हीजनवारण की अवधारणा क्वांटम भौतिकी के कई मौलिक सिद्धांतों के लिए महत्वपूर्ण है, जिसमें क्वांटम टनलिंग, क्वांटम बिहेवियर और क्वांटम जानकारी शामिल हैं। हीजनवारण के पीछे के सिद्धांतों को समझने से क्वांटम कंप्यूटिंग के क्षेत्र में प्रगति हो सकती है और क्वांटम यांत्रिकी के सिद्धांतों को बेहतर समझने में मदद मिल सकती है।|
| what does the word sophicticated mean?, answer in english then translate the answer  to hindi|(No specific input provided) | Sophisticated is a word that describes something that is very sophisticated or refined. It is often used to describe a person who is very cultured and has a high level of education and refined taste. It can also be used to describe a product or service that is of high quality and has been carefully crafted.  Translation: सौम्वरण एक ऐसा शब्द है जो किसी ऐसी चीज़ का वर्णन करता है जो बहुत ही सौम्वरण या परिष्कृत होती है। इसका उपयोग अक्सर किसी ऐसे व्यक्ति का वर्णन करने के लिए किया जाता है जो बहुत ही संस्कृत है और जिसकी उच्च स्तर की शिक्षा और परिष्कृत स्वाद है। इसका उपयोग किसी ऐसे उत्पाद या सेवा का वर्णन करने के लिए भी किया जा सकता है जो उच्च गुणवत्ता का हो और जिसे सावधानीपूर्वक तैयार किया गया हो।</s> |

# Dairava-Example output 
| Instruction | Input | Response |
|-------------|-------|----------|
| what does the word sophicticated mean?, answer in english then translate the answer to hindi | (No specific input provided) |Sophisticated means having a high level of refinement, sophistication, or sophistication. It is often used to describe someone who is well-educated, cultured, and refined in their speech and manner.|
| why is machine learning so hard?mujhe batado please? | (No specific input provided) | Machine learning is a complex and challenging field, and it can be difficult for beginners to understand the underlying principles and concepts. However, with practice and patience, anyone can learn the basics of machine learning and begin to develop their skills in this field. Some of the key challenges in machine learning include:1. Data: Machine learning algorithms require large amounts of high-quality data to train on, which can be difficult to obtain and prepare.2. Computational resources: Machine learning algorithms require a lot of computational power to run, which can be expensive and time-consuming to set up and maintain.3. Mathematics: Machine learning algorithms are based on complex mathematical concepts, such as linear algebra and probability theory, which can be difficult to grasp for beginners.4. Time and effort: Machine learning is a long-term process that requires a lot of time and effort to develop and refine.5. Interpretability: Machine learning models can be difficult to interpret, as they are often black boxes that are difficult to understand.6. Ethical considerations: Machine learning algorithms can have ethical implications, such as bias and discrimination, which can be difficult to navigate.7. Real-world applications: Machine learning is a rapidly evolving field, and the real-world applications of machine learning are constantly changing.8. Collaboration: Machine learning is a team sport, and collaboration is key to success. It can be difficult to work effectively with others on a team. इन चुनौतियों के बावजूद, मशीन लर्निंग एक रोमांचक और पुरस्कृत क्षेत्र है जो लगातार विकसित हो रहा है और नए अवसरों और चुनौतियों की ओर ले जा रहा है। With dedication and hard work, anyone can learn the basics of machine learning and begin to develop their skills in this field. |
| क्वांटम भौतिकी में हीजनवारण क्या है, और यह क्यों महत्वपूर्ण है? (What is superposition in quantum physics, and why is it important?)| (No specific input provided) | हीजनवारण एक ऐसी प्रक्रिया है जो एक ही कण को दो या दो से अधिक अलग-अलग अवस्थाओं में परिवर्तित करती है। यह एक मौलिक भौतिकी घटना है जो क्वांटम यांत्रिकी के सिद्धांतों के आधार पर होती है। हीजनवारण महत्वपूर्ण है क्योंकि यह क्वांटम यांत्रिकी के मौलिक सिद्धांतों को प्रदर्शित करता है, जैसे कि क्वांटम सुपरपोजिशन और क्वांटम उलझाव। हीजनवारण का उपयोग क्वांटम कंप्यूटर में क्वांटम बिट्स के निर्माण के लिए किया जाता है, जो क्वांटम कंप्यूटिंग के लिए एक महत्वपूर्ण तकनीक है। इसके अतिरिक्त, हीजनवारण का उपयोग क्वांटम टेलीपोर्टेशन के लिए किया जा सकता है, जो क्वांटम संचार के लिए एक महत्वपूर्ण तकनीक है।|


# Gaja-Prompt Format 
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

# Dairava-Prompt Format 

1) The prompt template for only instruction
```python
{system prompt}
<|user|>
{user_input}
<|assistant|>
```
# Merge configuration and info
- The merge method used here was the simple - Slerp
```yaml
slices:
  - sources:
      - model: ai4bharat/Airavata
        layer_range: [0, 32]
      - model: damerajee/Gaja
        layer_range: [0, 32]
merge_method: slerp
base_model: ai4bharat/Airavata
parameters:
  t:
    - filter: self_attn
      value: [0, 0.5, 0.3, 0.7, 1]
    - filter: mlp
      value: [1, 0.5, 0.7, 0.3, 0]
    - value: 0.5
dtype: bfloat16
```

# Local Inference 
You can try the model out  locally using Ollama
Steps to follow:
-  Download any of the gguf model you want from [Gaja.gguf](https://huggingface.co/damerajee/gaja-gguf)
-  install ollama either for Mac or Linux
-  Go to the folder Local inference and copy the Contents inside the Modelfile
-  create a txt file rename it Modelfile paste the Contents from inside the Modelfile to the text file you just created
-  Open your command prompt based on your device and os (make sure your in the same directory as where you saved your txt file and your gguf format model)
-  Type ollama create choose-a-model-name -f <location of the file e.g. ./Modelfile>'
-  ollama run choose-a-model-name
  For more Better Instruction please follow this docs -> [Ollama Modelfile](https://github.com/ollama/ollama/blob/main/docs/modelfile.md)

# Future plans
- Will be Instruct-fine tuning on more dataset( more than 100k + rows)
- Will be performing more merges using different techniques like ties , dare and more slerp
- Will be evaluating the model on both Hindi and English benchmark
  

# Usage Note
It's important to note that the models have not undergone detoxification. Therefore, while they possess impressive linguistic capabilities, there is a possibility for them to generate content that could be deemed harmful or offensive. We urge users to exercise discretion and supervise the model's outputs closely, especially in public or sensitive applications.
