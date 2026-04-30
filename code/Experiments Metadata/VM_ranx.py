# Import and variables' definition



###### load THE EXPERIMENT

import sys
import os
import pickle
import json
import copy
import pandas as pd
from tqdm import tqdm
#from google.colab import drive
# Mount Google Drive
#drive.mount('/content/drive')
import sys
import os

# import folder structure
from VM_00_01_folders_and_global_variables import project_dir,intermediate_results_dir,results_dir,prompts_dir,ranx_dir,info_g_variables,ranx_reports_dir

# Define name of generation text experiment
#g_exp_gen='g_gen_first_exp_scifact_S01' #$
g_exp_gen=info_g_variables['g_exp_gen']

# Define name of embeddings experiment
#g_exp_emb='g_emb_Mistral_S01' #$
g_exp_emb=info_g_variables['g_exp_emb']

# Define name of similarity experiment
#g_exp_sim='g_sim_Mistral_S01' #$
g_exp_sim=info_g_variables['g_exp_sim']

print(g_exp_gen, g_exp_emb, g_exp_sim)

# Load the dictionary with the metadata on similarity analysis with generated texts
## Similarity - Generated text - metadata
with open(os.path.join(results_dir,'infos_'+g_exp_sim+'.pkl'), "rb") as file:
    infos_g_exp_sim=pickle.load(file)

# Load the dictionary with the metadata on embeddings of generated texts
## Embeddings - Generated text - metadata
#with open(os.path.join(results_dir,'infos_'+infos_g_exp_sim['g_exp_emb']+'.pkl'), "rb") as file:
 #   infos_g_exp_emb=pickle.load(file)
with open(os.path.join(results_dir,'infos_'+g_exp_emb+'.pkl'), "rb") as file:
    infos_g_exp_emb=pickle.load(file)


# Load the dictionary with the metadata on generated text
## Generated text - metadata
#with open(os.path.join(results_dir,'infos_'+infos_g_exp_emb['g_exp_gen']+'.pkl'), "rb") as file:
 #   infos_g_exp_gen=pickle.load(file)
with open(os.path.join(results_dir,'infos_'+g_exp_gen+'.pkl'), "rb") as file:
  infos_g_exp_gen=pickle.load(file)

#some checks
print(infos_g_exp_sim['g_exp_emb']==g_exp_emb)
print(infos_g_exp_emb['g_exp_gen']==g_exp_gen)

"""# Ranx

## INSTALL AND LOAD
"""

os.system("pip install -U ranx")

from ranx import Qrels, Run, evaluate, compare, fuse, optimize_fusion
from f_load_beir import f_load_beir

# test set
corpus, queries_test, qrels_test = f_load_beir(ds = str(infos_g_exp_gen['Beir dataset']), split = 'train')
qrels_test_ranx = Qrels(qrels_test, name=str(infos_g_exp_gen['Beir dataset']+'_'+'test'))

#concatenate prompt keys and 'Original'
keys_to_use=list(infos_g_exp_gen['Prompts_keys'])
keys_to_use.append('Original')
print(keys_to_use)

# define data split (train, test, dev) to use for the analysis
split='test' #$

# Load the document scores for the relevant prompts, convert them to run objects and save them in a list
corpuses_scores_obj_ranx=[]

for item in keys_to_use:
  with open(os.path.join(results_dir,infos_g_exp_sim['g_exp_sim']+'_'+infos_g_exp_emb['g_exp_emb']+'_'+infos_g_exp_emb['g_exp_gen']+'_'+item+'_split_'+split+'_similarity_scores.pkl'), "rb") as file:
      item_similarity_load=pickle.load(file)
      corpuses_scores_obj_ranx.append(Run(item_similarity_load,name=item))

"""## Prompts and metrics"""

# Select different corpuses to fuse ranks
keys_rank_to_merge = info_g_variables['g_keys_rank_to_merge']

# Define metrics to compute
## except MAP and nDCG (separate sets, see belo)

metrics_to_use = []
prefixes = info_g_variables['g_metrics']

for prefix in prefixes:
  if '@' in prefix:
    for k in info_g_variables['g_k']:
        metrics_to_use.append(prefix + str(k))
  else: metrics_to_use.append(prefix)

# Let's investigate how some metrics behave with different k
## add MAP and nDCG @1:100 to the metrics list
metrics_to_use.extend(['map@'+str(k) for k in range(1, 101)])
metrics_to_use.extend(['ndcg@'+str(k) for k in range(1, 101)])

"""## Fusion - UNsupervised"""

# Define a list of unsupervised fusion algorithms to use
fusion_methods_unsupervised = info_g_variables['g_fusion_methods_unsupervised']

#### Compute the fused rankings according to the specified methods and generated texts combinations

# Define a normalization method
norm_method = 'min-max'


# objects to compare: e.g. ('EN1','EN2','EN3','EN4') and ('EX1','EX2')
# objects to fuse/combined run: e.g. ('EX1','EX2')


# Load the relevant document scores based on prompts ranks to fuse, convert them to run objects and save them in a list

objects_to_compare_ranx = [] #list to collect all run objects from the fusions

for fusion_method in tqdm(fusion_methods_unsupervised, desc='outer loop - fusion'):
  for item in tqdm(keys_rank_to_merge, desc='inner loop - load texts'): # 1.) # fuse the ranks for a specific combination of prompts
    objects_to_fuse_obj=[] # collect the objects to fuse
    for sub_item in item: # 2.) compute for all elements of a specific combination of prompts the Run object
      for obj_ranx in corpuses_scores_obj_ranx:
        if obj_ranx.name == sub_item:
          objects_to_fuse_obj.append(obj_ranx)
    combined_run_objs = fuse(
        runs=objects_to_fuse_obj,
        norm=norm_method,
        method=fusion_method
    )
    combined_run_objs.name=fusion_method+'_'+'|'.join(item)
    # save the list with the objects to compare at the end of each fusion round
    file_path=os.path.join(ranx_dir,'combined_run_objs_'+infos_g_exp_sim['g_exp_sim']+'_split_'+split+'_'+combined_run_objs.name+'.pkl') #?
    combined_run_objs_dict=combined_run_objs.to_dict() #convert to dictionary
    with open(file_path, "wb") as file:
      pickle.dump(combined_run_objs_dict, file)

    objects_to_compare_ranx.append(combined_run_objs)

  # in case the execution of the loops stop
  file_path_txt = os.path.join(results_dir,'StopLoop-objects_to_compare_ranx.pkl')
  msg=fusion_method+'_'+'_'.join(item)+'_'+sub_item
  with open(file_path_txt, "w") as file:
      # Write the string to the file
      file.write(msg)

# Apped the files for the single prompts
for item in keys_to_use:
  with open(os.path.join(results_dir,infos_g_exp_sim['g_exp_sim']+'_'+infos_g_exp_emb['g_exp_emb']+'_'+infos_g_exp_emb['g_exp_gen']+'_'+item+'_split_'+split+'_similarity_scores.pkl'), "rb") as file: #?
      item_similarity_load=pickle.load(file)
      objects_to_compare_ranx.append(Run(item_similarity_load,name='NoFusion_'+item))

'''# [!!! IF NECESSARY] load from disk objects_to_compare_ranx
objects_to_compare_ranx = [] #list to collect all run objects from the fusions

for fusion_method in tqdm(fusion_methods_unsupervised, desc='outer loop - fusion'):
  for item in tqdm(keys_rank_to_merge, desc='inner loop - load texts'): # 1.) # fuse the ranks for a specific combination of prompts
    file_path=os.path.join(ranx_dir,'combined_run_objs_'+infos_g_exp_sim['g_exp_sim']+'_split_'+split+'_'+fusion_method+'_'+'|'.join(item)+'.pkl') #?
    #with open(file_path, 'r') as f:
      # Load the JSON data and create a Run object
     # obj_to_load=json.load(f)
    with open(file_path, 'rb') as file:
      object_loaded = pickle.load(file)
    # Create a Run file of the loaded object
    object_loaded_run=Run(object_loaded, name=fusion_method+'_'+'|'.join(item))

    objects_to_compare_ranx.append(object_loaded_run)

# Apped the files for the single prompts
for item in keys_to_use:
  with open(os.path.join(results_dir,infos_g_exp_sim['g_exp_sim']+'_'+infos_g_exp_emb['g_exp_emb']+'_'+infos_g_exp_emb['g_exp_gen']+'_'+item+'_split_'+split+'_similarity_scores.pkl'), "rb") as file: #?
      item_similarity_load=pickle.load(file)
      objects_to_compare_ranx.append(Run(item_similarity_load,name='NoFusion_'+item))
'''

### REPORT FUSION UNSUPERVISED

os.system("pip install openpyxl")
# In the Ranx report there is somehow a limit (27) on the elements (rows) which can be compared in a Report object.
# Thus the results are to be split in packages and reassembled in a dataframe together

step = 27
#metrics_to_use = ['hit_rate@10', 'ndcg@10']
#metrics_to_use = ['hit_rate@1','hit_rate@10']


# Initialize an empty DataFrame
df_report = pd.DataFrame(columns=['model_name','Embeddings','fusion','type','prompts'] + metrics_to_use)

# Loop through objects to compare_ranx with a step size
for i in range(0, len(objects_to_compare_ranx), step):
  # Create a sublist of objects to compare
  temp_objects_to_compare_ranx = objects_to_compare_ranx[i:i+step]

  # Create report with the subset of objects to compare
  temp_report = compare(
      qrels_test_ranx,
      metrics=metrics_to_use,
      rounding_digits=6,
      runs=temp_objects_to_compare_ranx
  )

  #msg in case the loop stops
  msg='ok'
  # Convert temp_report to dictionary
  temp_report_dict = temp_report.to_dict()

  # Iterate over each model in the temp_report_dict
  for element_name, model_data in temp_report_dict.items():
      # Check if the model name is in the list of model names
    if element_name in temp_report_dict['model_names']:
        # Extract scores for the model
        scores = model_data['scores']
        # new data to add to the report dataframe
        new_data = {'model_name': element_name,'Embeddings':infos_g_exp_emb['g_embedding_model'],'fusion':element_name.rsplit('_',1)[0],'type':'unsupervised','prompts':element_name.rsplit('_',1)[1], **scores}
        # convert the dictionary to a DataFrame
        new_row = pd.DataFrame([new_data])

        # Reset index of df_report to ensure it's unique
        #df_report.reset_index(drop=True, inplace=True)

        # Concatenate df_report with new_row along axis 0 (rows)
        df_report = pd.concat([df_report, new_row], ignore_index=True)
        #msg in case the loop stops
        msg = str(i)+'_'+element_name
  # save the report
  file_path_report_unsupervised=os.path.join(ranx_reports_dir,'df_report_'+infos_g_exp_sim['g_exp_sim']+'_fusion_unsupervised.pkl')
  with open(file_path_report_unsupervised, "wb") as file:
    pickle.dump(df_report, file) #pkl file
  df_report.to_excel(file_path_report_unsupervised.rsplit(".", 1)[0] + ".xlsx", index=False) #excel file
  # in case the execution of the loops stops
  file_path_txt = os.path.join(results_dir,'StopLoop-objects_to_compare_ranx.pkl')
  msg=fusion_method+'_'+ str(i)
  with open(file_path_txt, "w") as file:
      # Write the string to the file
      file.write(msg)



