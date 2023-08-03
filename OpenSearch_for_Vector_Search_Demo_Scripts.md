# OpenSearch 2.7 for Vector Search
_[Experimental Local model support using OpenSearch Neural plugin available with version 2.7]_
## _Power your retail store with semantic search in just minutes!_


#### Step1 Reset the only_run_on_ml_node to false

In dev tools , we first Reset the default plugins.ml_commons.only_run_on_ml_node to false
Neural Search plugin is an experimental feature and does not support ML nodes with GPU instances just yet. 
So, we run our models on our data node by setting the "only_run_on_ml_node": false
Once ML nodes are launched on the OpenSearch Service, you can take advantage of GPU acceleration on your ML node

```json
PUT _cluster/settings
{
   "persistent":{
     "plugins.ml_commons.only_run_on_ml_node": false
   }
}
```
The _GET_ __cluster/settings_ will allow you validate if the _only_run_on_ml_node_ configuration is set to false.
```json
GET _cluster/settings
```

#### Step2 - Upload a pre-trained model to OpenSearch cluster
The demo uses the Paraphrase-multilingual-MiniLM- model available in the OpenSearch documentation. But checkout the [OpenSearch documentation](https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/) for other pre-trained models.
Model-serving framework that we saw earlier supports text embedding models.
As of Version 2.5, OpenSearch only supports the TorchScript and ONNX formats.
Model size , model_content_size_in_bytes is also provided.  Most deep learning models are more than 100 MB. For this reason OpenSearch splits the model file into smaller chunks to be stored in a model index. So, make sure you correctly size your ML nodes so that you have enough memory when making ML inferences. For this demo, I have used 2 r6gd.4xlarge nodes but this really depends on the size of the dataset and the model you plan to upload.
Model config includes the _model_type_, _embedding_dimension_, _framework_type_ and the _all_config_.
The _all_config_ field is used for reference purposes, You can specify all model configurations in this field. Once the model is uploaded, you can use the _GET /_plugins/_ml/models/_ API to get all model configurations stored in this field.
_url_, the model file must be saved as zip files before upload.

```json
POST /_plugins/_ml/models/_upload
{
    "name": "huggingface/sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
    "version": "1.0.1",
    "model_format": "TORCH_SCRIPT"
}
```

Once you execute the above command, Get the task id from the response. Using task id , get the model id.

```json
GET /_plugins/_ml/tasks/<TaskID>
```

#### Step 3 - Load the model from the model index. 
The load model operation reads the model’s chunks from the model index and then creates an instance of the model to load into memory. The bigger the model, more will be the number of chunks generated. And More the number of chunks, longer it would take to load the model into the memory. 
```json
POST /_plugins/_ml/models/<modelId>/_load
```

Once the _load command is executed, Get the task id from the  response
Check the load status by task id , this will confirm that the model is loaded into the memory.
```json
GET /_plugins/_ml/tasks/<TaskID>
```

#### Step 4 - Once the load is completed, use the model id to create a pipeline

As you can see the pipeline includes the modelid and 
Defines the 2 fields from the retail product catalog for which vectors are generated

```json
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
```

#### Step 5 - Create your index and attach it to the Neural Search pipeline.
Because the index maps to k-NN vector fields, the index setting field index-knn is set to true. For the fields defined as vertors in your pipeline, use k-NN method definitions to specify type, dimension and method.

```json
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
```

#### Step 6 - Ingest the retail product catalog to your new semantic_demostore. 
You could either reidnex your existing retail demo store with the mapping for vector embedding. Or you could also _bulk ingest to the new index that includes the pre-defined vector fields. For the demo, we will __reindex_ from the existing retail product catalog.

```json
POST _reindex
{
  "source": {
    "index": "demostore"
  },
  "dest": {
    "index": "semantic_demostore"
  }
}
```

_Congratulations! The index is now created!_ The semantic_demostore index will include the vector embeddings for the name and description field.


Now lets move over the OpenSearch Dashboard and compare the difference between a Vanilla search and Semantic Neural Search . We will be using the "Search Relevance plugin to compare the results.

On the left we have the _Vanilla Search_ DSL query 
```json
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
```


On the right we have the _Neural Search_ DSL query

```json
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
```

_Now have fun trying out [OpenSearch Neural Plugin](https://opensearch.org/docs/latest/search-plugins/neural-search/) to build vector search using different pre-trained models on the [OpenSearch website](https://opensearch.org/docs/latest/ml-commons-plugin/pretrained-models/)!_
