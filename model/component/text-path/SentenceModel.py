from sentence_transformers import SentenceTransformer

def SentenceModel():
    return SentenceTransformer("/public1/cjh/workspace/DepressionPrediction/model/pretrained_model/all-MiniLM-L12-v1",device="cuda:1")
