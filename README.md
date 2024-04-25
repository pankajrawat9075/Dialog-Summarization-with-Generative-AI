# Dialog-Summarization-with-Generative-AI

Using Open-Source LLMs like FLAN-T5, built a Dialog Summarization model and did fine-tuning with DialogSum HF Dataset.

**Tasks completeted:**
# 1. Generative AI Use Case: Summarize Dialogue

In this lab we did the dialogue summarization task using generative AI. We explored how the input text affects the output of the model, and performed prompt engineering to direct it towards the task we need. By comparing zero shot, one shot, and few shot inferences, we took the first step towards prompt engineering and see how it can enhance the generative output of Large Language Models.

## Table of Contents
- 1 - Set up Kernel and Required Dependencies
- 2 - Summarize Dialogue without Prompt Engineering
- 3 - Summarize Dialogue with an Instruction Prompt
  - 3.1 - Zero Shot Inference with an Instruction Prompt
  - 3.2 - Zero Shot Inference with the Prompt Template from FLAN-T5
- 4 - Summarize Dialogue with One Shot and Few Shot Inference
  - 4.1 - One Shot Inference
  - 4.2 - Few Shot Inference
- 5 - Generative Configuration Parameters for Inference

## Results:
```
Example 1

INPUT PROMPT:
  
Summarize the following conversation. Try to reason on the conversation and then reply the summary.  
  
#Person1#: What time is it, Tom?  
#Person2#: Just a minute. It's ten to nine by my watch.  
#Person1#: Is it? I had no idea it was so late. I must be off now.  
#Person2#: What's the hurry?  
#Person1#: I must catch the nine-thirty train.  
#Person2#: You've plenty of time yet. The railway station is very close. It won't take more than twenty minutes to get there.  
  
Summary:  
  
---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# is in a hurry to catch a train. Tom tells #Person1# there is plenty of time.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
The train is about to leave.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT - with FLAN-T5 Prompt Template:
Tom is late for the train.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - ONE SHOT:
Tom is late for the train. He has to catch it at 9:30.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - FEW SHOT:
Tom is late for the train. He has to catch it at 9:30.
```



# 2. Fine-Tune a Generative AI Model for Dialogue Summarization with AWS Sagemaker

We then fine-tuned an existing LLM from Hugging Face for enhanced dialogue summarization. We used the [FLAN-T5](https://huggingface.co/docs/transformers/model_doc/flan-t5) model, which provides a high quality instruction tuned model and can summarize text out of the box. To improve the inferences, we explored a full fine-tuning approach and evaluated the results with ROUGE metrics. Then we performed Parameter Efficient Fine-Tuning (PEFT), evaluated the resulting model and saw that the benefits of PEFT outweigh the slightly-lower performance metrics.

## Table of Contents
- 1 - Set up Kernel, Load Required Dependencies, Dataset and LLM
  - 1.1 - Set up Kernel and Required Dependencies
  - 1.2 - Load Dataset and LLM
  - 1.3 - Test the Model with Zero Shot Inferencing
- 2 - Perform Full Fine-Tuning
  - 2.1 - Preprocess the Dialog-Summary Dataset
  - 2.2 - Fine-Tune the Model with the Preprocessed Dataset
  - 2.3 - Evaluate the Model Qualitatively (Human Evaluation)
  - 2.4 - Evaluate the Model Quantitatively (with ROUGE Metric)
- 3 - Perform Parameter Efficient Fine-Tuning (PEFT)
  - 3.1 - Setup the PEFT/LoRA model for Fine-Tuning
  - 3.2 - Train PEFT Adapter
  - 3.3 - Evaluate the Model Qualitatively (Human Evaluation)
  - 3.4 - Evaluate the Model Quantitatively (with ROUGE Metric)

## Results:
```
Example
---------------------------------------------------------------------------------------------------
INPUT PROMPT:

Summarize the following conversation.

#Person1#: Have you considered upgrading your system?
#Person2#: Yes, but I'm not sure what exactly I would need.
#Person1#: You could consider adding a painting program to your software. It would allow you to make up your own flyers and banners for advertising.
#Person2#: That would be a definite bonus.
#Person1#: You might also want to upgrade your hardware because it is pretty outdated now.
#Person2#: How can we do that?
#Person1#: You'd probably need a faster processor, to begin with. And you also need a more powerful hard disc, more memory and a faster modem. Do you have a CD-ROM drive?
#Person2#: No.
#Person1#: Then you might want to add a CD-ROM drive too, because most new software programs are coming out on Cds.
#Person2#: That sounds great. Thanks.

Summary:

---------------------------------------------------------------------------------------------------
BASELINE HUMAN SUMMARY:
#Person1# teaches #Person2# how to upgrade software and hardware in #Person2#'s system.
---------------------------------------------------------------------------------------------------
MODEL GENERATION - ZERO SHOT:
#Person1#: I'm thinking of upgrading my computer.
---------------------------------------------------------------------------------------------------
ORIGINAL MODEL:
#Person1#: You'd like to upgrade your computer. #Person2: You'd like to upgrade your computer.
---------------------------------------------------------------------------------------------------
INSTRUCT MODEL - FULL-FINETUNING
#Person1# suggests #Person2# upgrading #Person2#'s system, hardware, and CD-ROM drive. #Person2# thinks it's great.
---------------------------------------------------------------------------------------------------
PEFT MODEL: #Person1# recommends adding a painting program to #Person2#'s software and upgrading hardware. #Person2# also wants to upgrade the hardware because it's outdated now.
```
```
ROUGE METRICS over FULL DATASET - used to evaluate summaries - higher is better

ORIGINAL MODEL:
{'rouge1': 0.2334158581572823, 'rouge2': 0.07603964187010573, 'rougeL': 0.20145520923859048, 'rougeLsum': 0.20145899339006135}
INSTRUCT MODEL:
{'rouge1': 0.42161291557556113, 'rouge2': 0.18035380596301792, 'rougeL': 0.3384439349963909, 'rougeLsum': 0.33835653595561666}
PEFT MODEL:
{'rouge1': 0.40810631575616746, 'rouge2': 0.1633255794568712, 'rougeL': 0.32507074586565354, 'rougeLsum': 0.3248950182867091}


Absolute percentage improvement of INSTRUCT MODEL over ORIGINAL MODEL
rouge1: 18.82%
rouge2: 10.43%
rougeL: 13.70%
rougeLsum: 13.69%


Absolute percentage improvement of PEFT MODEL over ORIGINAL MODEL
rouge1: 17.47%
rouge2: 8.73%
rougeL: 12.36%
rougeLsum: 12.34%


Absolute percentage improvement of PEFT MODEL over INSTRUCT MODEL
rouge1: -1.35%
rouge2: -1.70%
rougeL: -1.34%
rougeLsum: -1.35%
```

# 3. Fine-Tune FLAN-T5 with Reinforcement Learning (PPO) and PEFT to Generate Less-Toxic Summaries

We then fine-tuned a FLAN-T5 model to generate less toxic content with Meta AI's hate speech reward model. The reward model is a binary classifier that predicts either "not hate" or "hate" for the given text. We used Proximal Policy Optimization (PPO) to fine-tune and reduce the model's toxicity.

## Table of Contents
- 1 - Set up Kernel and Required Dependencies
- 2 - Load FLAN-T5 Model, Prepare Reward Model and Toxicity Evaluator
  - 2.1 - Load Data and FLAN-T5 Model Fine-Tuned with Summarization Instruction
  - 2.2 - Prepare Reward Model
  - 2.3 - Evaluate Toxicity
- 3 - Perform Fine-Tuning to Detoxify the Summaries
  - 3.1 - Initialize `PPOTrainer`
  - 3.2 - Fine-Tune the Model
  - 3.3 - Evaluate the Model Quantitatively
  - 3.4 - Evaluate the Model Qualitatively
 ## Results:
```
Percentage improvement of toxicity score after detoxification:
mean: -13.66%
std: -6.60%
```

