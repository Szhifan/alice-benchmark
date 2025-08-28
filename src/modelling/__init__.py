from .custom_llama import LlamaModel, LlamaForSequenceClassification
from .modelling_utils import *
import transformers.models


transformers.models.llama.modeling_llama.LlamaModel = LlamaModel
transformers.models.llama.modeling_llama.LlamaForSequenceClassification = LlamaForSequenceClassification