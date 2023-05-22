 
## Here we are in dev tools 
## As of Step1 , we first Reset the default plugins.ml_commons.only_run_on_ml_node to false
## Neural Search plugin is an experimental feature and does not support ML nodes with GPU instances just yet. 
## So, we run our models on our data node by setting the "only_run_on_ml_node": false
## Once ML nodes are launched on the OpenSearch Service, you can take advantage of GPU acceleration on your ML node
PUT _cluster/settings
{
   "persistent":{
     "plugins.ml_commons.only_run_on_ml_node": false
   }
}

## the Get _cluster/settings will allow you validate if this configuration is set to false.
GET _cluster/settings



## Step2 - Lets upload a pre-trained model
## I have used the Paraphrase-multilingual-MiniLM- model available in the OpenSearch documentation

## Model-serving framework that we saw earlier supports text embedding models.
## As of Version 2.5, OpenSearch only supports the TorchScript and ONNX formats.

## Model size , model_content_size_in_bytes is also provided.  Most deep learning models are more than 100 MB , 
### For this reason OpenSearch splits the model file into smaller chunks to be stored in a model index 
### So, make sure you correctly size your ML nodes so that you have enough memory when making ML inferences.
## For this demo, I have used 2 r6gd.4xlarge nodes but this really depends on the size of the dataset and the model you plan to upload.

## Model config    includes the model_type, embedding_dimension, framework_type and the all_config.

## The all_config field is used for reference purposes, You can specify all model configurations in this field. Once the model is uploaded, you can use the get-model-API operation to get all model configurations stored in this field.

## URL - The model file must be saved as zip files before upload.

POST /_plugins/_ml/models/_upload
{
    "name": "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "version": "1.0.1",
    "description": "his is a sentence-transformers model: It maps sentences & paragraphs to a 384 dimensional dense vector space and can be used for tasks like clustering or semantic search",
    "model_task_type": "TEXT_EMBEDDING",
    "model_format": "TORCH_SCRIPT",
    "model_content_size_in_bytes": 488135181,
    "model_content_hash_value": "a2ae3c4f161bd8e5a99a19ba5589443d33a120bb2bd67aa9da102c8b201f1277",
    "model_config": {
        "model_type": "bert",
        "embedding_dimension": 384,
        "framework_type": "sentence_transformers",
        "all_config": "{\"_name_or_path\":\"old_models/paraphrase-multilingual-MiniLM-L12-v2/0_Transformer\",\"architectures\":[\"BertModel\"],\"attention_probs_dropout_prob\":0.1,\"gradient_checkpointing\":false,\"hidden_act\":\"gelu\",\"hidden_dropout_prob\":0.1,\"hidden_size\":384,\"initializer_range\":0.02,\"intermediate_size\":1536,\"layer_norm_eps\":1e-12,\"max_position_embeddings\":512,\"model_type\":\"bert\",\"num_attention_heads\":12,\"num_hidden_layers\":12,\"pad_token_id\":0,\"position_embedding_type\":\"absolute\",\"transformers_version\":\"4.7.0\",\"type_vocab_size\":2,\"use_cache\":true,\"vocab_size\":250037}"
    },
    "created_time": 1676326534702,
    "url": "https://artifacts.opensearch.org/models/ml-models/huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2/1.0.1/torch_script/sentence-transformers_paraphrase-multilingual-MiniLM-L12-v2-1.0.1-torch_script.zip"
}


## Once you execute the above command, Get the Task ID from the response
## Task ID = 

## Using Task ID , get the model ID 
GET /_plugins/_ml/tasks/<TaskID>


## Step 3 - Load the model from the model index. 
## The load model operation reads the model’s chunks from the model index and then creates an instance of the model to load into memory. The bigger the model, more will be the number of chunks generated.
## AND More the number of chunks, longer it would take to load the model into the memory. 
POST /_plugins/_ml/models/<modelId>/_load

## Once the _load command is executed, Get the Task ID from the  response
## Task ID = 

# Check the load status by task id , this will confirm that the model is loaded into the memory.
GET /_plugins/_ml/tasks/<TaskID>

## Step 4 - Once the load is completed, use the model id to create a pipeline

### As you can see the pipeline includes the modelid and 
## Defines the 2 fields from the retail product catalog for which vectors are generated
PUT _ingest/pipeline/neural-pipeline
{
  "description": "Semantic Search for retail product catalog ",
  "processors" : [
    {
      "text_embedding": {
        "model_id": "modelId",
        "field_map": {
           "description": "desc_v",
           "name": "name_v"
        }
      }
    }
  ]
}

## Step 5 - Create your index and attache it to the Neural Search pipeline.
## Because the index maps to k-NN vector fields, the index setting field index-knn is set to true

## For the fields defined as vertors in your pipeline, use k-NN method definitions to specify type, dimension and method.

PUT semantic_demostore
{
  "settings": {
    "index.knn": true,  
    "default_pipeline": "neural-pipeline",
    "number_of_shards": 1,
    "number_of_replicas": 1
  },
  "mappings": {
    "properties": {
      "desc_v": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "engine": "nmslib",
          "space_type": "cosinesimil"
        }
      },
      "name_v": {
        "type": "knn_vector",
        "dimension": 384,
        "method": {
          "name": "hnsw",
          "engine": "nmslib",
          "space_type": "cosinesimil"
        }
      },
      "description": {
        "type": "text" 
      },
      "name": {
        "type": "text" 
      } 
    }
  }
}


## Step 6 - You could either reidnex your existing retail demo store with the mapping for vector embedding
## Or you could also _bulk ingest to the new index that includes the pre-defined vector fields.

POST _reindex
{
  "source": {
    "index": "demostore"
  },
  "dest": {
    "index": "semantic_demostore"
  }
}

## Now you are done ! The semantic_demostore will include the vector embeddings for the name and description field.


## Now lets move over the OpenSearch Dashboard and compare the difference between a Vanilla search and Semantic Neural Search . We will be using the "Search Relevance plugin to compare the results.

## On the left we have the Vanilla search 
{
  "size": 100, 
  "_source": {
    "includes": ["name", "description"]
  },
  "query": {
    "multi_match": {
      "query": "%SearchText%",
      "fields": ["name", "description"]
    }
  }
}

## On the right we have the Neural Search
{
  "size": 100, 
  "_source": {
    "includes": ["name", "description"]
  },
  "query": {
    "neural": {
      "name_v": {
        "query_text": "%SearchText%",
        "model_id": "modelId",
        "k": 100
      }
    }
  }
}

