import json
from models.SDEC_AD import DeepEmbeddingClustering
from scriptss.load_data import load_data
import sys

if __name__ == "__main__":
    fn17_embedding_file = "mydata/doc2vec_triple/triple_doc2vec_first.p"
    fnplus_embedding_file = "mydata/doc2vec_triple/triple_doc2vec_second.p"
    dict1_file = "dicts/text_num_type.json"
    with open(dict1_file, 'r', encoding='utf-8') as df1:
        dict1 = json.loads(df1.readline())
        df1.close()
    newX, newY, num_types, cut_off = load_data(fn17_embedding_file, fnplus_embedding_file, dict1)
    c = DeepEmbeddingClustering(n_clusters=num_types,
                                input_dim=300,
                                encoders_dims=[7500, 1000])

    ##### Training #####
    print("Train Autoencoder...")
    c.initialize(newX[:cut_off],
                 y=newY[:cut_off],
                 finetune_iters=50000,
                 layerwise_pretrain_iters=25000,
                 save_autoencoder=True)

    print("Clustering...")
    L = c.cluster(newX, y=newY, cut_off=cut_off, iter_max=1e6)



