#Nota Bene: stop tokens are for llama3. For Zephyr 7b delete the input "stop"
# INPUTS:
## corpus_docs: the documents from which the texts have to be generated. They have to be passed to the function in the same format as a corpus from the Beir dataset
## prompt: a SINGLE prompt to generate text (eg: "Can you summarize this text")

from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,ranx_dir,info_g_variables
from openai import OpenAI


def f_beir_gen_text(corpus_docs,prompt):
  
  client = OpenAI(base_url=f"http://localhost:19004/v1", api_key="EMPTY")
  try:

    prompt_key=list(prompt.keys())[0]

    corpus_experiment_prompt={} # dictionary to collect generated text for a SPECIFIC prompt and each document in the corpus

    for d in corpus_docs.keys():
      full_prompt = f"{prompt[prompt_key]}{corpus_docs[d]['text']}"
      response = client.chat.completions.create(
      model=info_g_variables['g_llm'],
      messages= [{"role": "user", "content": full_prompt}],
      temperature=info_g_variables['temperature'],
      max_tokens=2048#,
#      stop=[
#              "<|start_header_id|>",
#             "<|end_header_id|>",
#              "<|eot_id|>",
#             "<|reserved_special_token|>"
 #         ]
          #request_timeout=1
      )
      d_gen_text=response.choices[0].message.content

      corpus_experiment_prompt.update({d:{'text':d_gen_text,'title':corpus_docs[d]['title']}})
  except Exception as e:
    print(f"Error: {e}")
    return None
  
  client.close()
  return corpus_experiment_prompt