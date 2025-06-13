# Week 3-5: Understanding Sequences with Recurrent Neural Networks (RNN), Transformers and GenAI Basics

Welcome to Week 3-4 of our journey! This week, we'll delve into the fascinating world of Recurrent Neural Networks (RNNs) and explore how they can be used to process sequences of data, such as sentences and text.

### Recurrent Neural Network

## Coursera Course

### **Strongly Recommended**

**Complete the first two weeks of the course ["Sequence Models" by Andrew Ng](https://www.coursera.org/learn/nlp-sequence-models?) on Coursera.**

This course covers essential concepts related to sequential neural networks and word embeddings. You can audit the content for free, providing a solid theoretical foundation for the topics we'll cover this week.

## Understanding RNNs

Recurrent Neural Networks (RNNs) are designed to process sequences of data. Learn the basics of RNNs and how they maintain a memory of past inputs:

- [**Understanding RNNs - Towards Data Science**](https://towardsdatascience.com/illustrated-guide-to-recurrent-neural-networks-79e5eb8049c9)
- [**Recurrent Neural Networks (RNN) - Deep Learning Basics**](https://www.youtube.com/watch?v=UNmqTiOnRfg)

## Python Classes

Understanding Python classes is crucial for implementing and organizing code effectively. Classes provide a blueprint for creating objects, bundling data, and functionality together. Here are some resources for a quick introduction:

- [**Short Introduction to Python Classes**](https://www.geeksforgeeks.org/python-classes-and-objects/)
- [**PyTorch Docs - Implementing models using classes**](https://pytorch.org/tutorials/beginner/introyt/modelsyt_tutorial.html)

## Optimizers

Optimizers are pivotal for training deep learning models. They adjust model parameters during training to minimize the error or loss function. Familiarize yourself with common optimizers like SGD and Adam:

- [**PyTorch Docs - Optimizers**](https://pytorch.org/docs/stable/optim.html)
- [**Explanation of Common Optimizers**](https://towardsdatascience.com/optimizers-for-training-neural-network-59450d71caf6)
- [**CS231n Material on Optimization**](https://cs231n.github.io/neural-networks-3/#update)

These resources will prepare you for the exciting world of sequence processing with RNNs. Stay curious and enjoy your learning journey!


## Coursera Course

To dive deeper into the concepts covered, it's highly recommended to complete the remaining two weeks of the course "Sequence Models" by Andrew Ng on Coursera. You can audit the course for free, and it provides valuable insights into attention mechanisms, transformers, and large language models.

[**Sequence Models on Coursera**](https://www.coursera.org/learn/nlp-sequence-models?)

## Transformers

Transformers represent a breakthrough in neural networks for sequence processing, offering a powerful approach to understanding context in sequences. The following resources will help you grasp the fundamental concepts behind transformers:

- [**The Illustrated Transformer**](https://towardsdatascience.com/illustrated-guide-to-transformers-step-by-step-explanation-f74876522bc0): A visually informative guide to understanding how transformers work.
- [**StatQuest Video on Transformers**](https://www.youtube.com/watch?v=zxQyTK8quyY): An engaging video explaining transformer concepts.
- For better visualisation of LLMs and Transformers - Watch videos 5 to 8 from [this playlist](https://www.youtube.com/playlist?list=PLZHQObOWTQDNU6R1_67000Dx_ZCJB-3pi)
  
Optional Videos (Advanced) - Can complete after the project as these will be very helpful in understanding Transformers
- [**Attention is All You Need - Paper review**](https://www.youtube.com/watch?v=bCz4OMemCcA)
- [**Let's build GPT: from scratch, in code, spelled out.**](https://www.youtube.com/watch?v=kCc8FmEb1nY)


### Large Language Models (LLMs)

Large Language Models, often based on transformer architectures, have revolutionized natural language processing. They excel in understanding language intricacies, thanks to self-attention mechanisms. Explore more about LLMs with these resources:

- [**Introduction to Large Language Models**](https://research.aimultiple.com/large-language-models/): An introductory guide to understanding the significance of LLMs.
- [**Geeks for Geeks Article on LLM**](https://www.geeksforgeeks.org/large-language-model-llm/): In-depth information on Large Language Models.


### Decoding Strategies

Decoding strategies play a crucial role in generating meaningful sequences from LLMs. Learn about various decoding strategies to refine model outputs:

- [**Hugging Face Blog on Decoding**](https://huggingface.co/blog/how-to-generate): Insights into decoding strategies.
- [**Medium Article on Decoding Strategies**](https://towardsdatascience.com/decoding-strategies-that-you-need-to-know-for-response-generation-ba95ee0faadc): A concise guide with helpful graphs.


## Hugging Face 🤗

Hugging Face is a prominent community in natural language processing, providing tools and resources for researchers and developers. Their **Transformers** library is central to accessing and fine-tuning pre-trained language models.


### 🤗 Transformers

Explore the extensive capabilities of the **Transformers** library:

- [**🤗 Transformers Documentation**](https://huggingface.co/docs/transformers/index): Comprehensive documentation for the Transformers library.



### UI Options: Streamlit and Gradio

For creating interactive user interfaces, consider using **Streamlit** or **Gradio**. Both are user-friendly options for showcasing models.
Gradio is simpler than Streamlit. 

- **Streamlit:**
  - [**Streamlit Documentation**](https://docs.streamlit.io/): Learn to create interactive web apps with Streamlit.

- **Gradio:**
  - [**Gradio Documentation**](https://www.gradio.app/docs/interface): Explore Gradio's documentation for building intuitive interfaces.


## Transfer Learning and Fine-tuning

Transfer learning and fine-tuning are essential techniques to adapt pre-trained models to specific tasks. Here are resources to guide you through these processes:

- [**Overview of Transfer Learning**](https://medium.com/@atmabodha/pre-training-fine-tuning-and-in-context-learning-in-large-language-models-llms-dd483707b122): A brief theoretical overview.
- [**Stepwise Guide to Fine-tuning LLMs**](https://www.simform.com/blog/completeguide-finetuning-llm/#:~:text=Fine%2Dtuning%20in%20large%20language,your%20specific%20business%20use%20cases.): A step-by-step overview.
- [**Future of Natual Language Processing**](https://www.youtube.com/watch?v=G5lmya6eKtc)(_Optional_)

For more advanced users interested in building models from scratch, explore [**this YouTube video on building a GPT-style model**](https://youtu.be/kCc8FmEb1nY) (Note: Advanced content beyond the scope of this course). Proceed with caution.

Now you are well-equipped to explore the world of transformers, LLMs, and fine-tuning techniques.


## Retrieval Augmented Generation (RAG)

RAG is an AI framework for retrieving facts from an external knowledge base to ground large language models (LLMs) on the most accurate, up-to-date information and to give users insight into LLMs' generative process.

- [**Article on RAG**](https://research.ibm.com/blog/retrieval-augmented-generation-RAG)
- [**Vector Database Embeddings**](https://www.deeplearning.ai/short-courses/vector-databases-embeddings-applications/)
- [**Advanced retrieval for AI**](https://www.deeplearning.ai/short-courses/advanced-retrieval-for-ai/)(Optional)
- Research more about RAG from articles and youtube videos.


## Langchain

LangChain is a framework designed to simplify the creation of applications using large language models. As a language model integration framework, LangChain's use-cases largely overlap with those of language models in general, including document analysis and summarization, chatbots, and code analysis

- [**Documentation**](https://python.langchain.com/docs/get_started/introduction)
- [**Video Turorial**](https://youtu.be/_FpT1cwcSLg?si=TA6yxvoe9MZRR0IB)

Watch these two courses instead of the above video for better understanding of Langchain(Optional but Recommended)
- [**Langchain for LLM application Development**](https://www.deeplearning.ai/short-courses/langchain-for-llm-application-development/)
- [**Langchain Chat with your data**](https://www.deeplearning.ai/short-courses/langchain-chat-with-your-data/)


Go through the documentation(_Recommended_) or the video tutorial(_Simpler to understand_) of langchain and explore how can RAG be implemented using it.

