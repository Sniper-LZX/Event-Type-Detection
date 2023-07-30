from models.SDEC_AD import DeepEmbeddingClustering
from scriptss.load_data import load_data
from evaluation.clustering import external_eval_clusters, print_external_eval_clusters
import json

if __name__ == "__main__":
    fn17_embedding_file = "mydata/doc2vec_triple/triple_doc2vec_first.p"
    fnplus_embedding_file = "mydata/doc2vec_triple/triple_doc2vec_second.p"
    weights = "trained_SDEC_AD(DuEE-fin)/doc2vec_Triple/SDEC_AD_bcubed_fscore_0.11535.h5"
    dict1_file = "dicts/text_num_type.json"
    with open(dict1_file, 'r', encoding='utf-8') as df1:
        dict1 = json.loads(df1.readline())
        df1.close()

    newX, newY, num_frames, cut_off = load_data(fn17_embedding_file, fnplus_embedding_file, dict1)
    c = DeepEmbeddingClustering(n_clusters=num_frames,
                                input_dim=300,
                                encoders_dims=[7500, 1000])

    ##### Prediction #####
    print("Predicting...")
    # use the saved model when running the clustering (c.cluster)

    pred_Y = c.predict(newX[:cut_off], SDEC_trained_weights=weights)

    ##### Evaluation #####
    print("Evaluating...")
    print_external_eval_clusters(*external_eval_clusters(newY, pred_Y))
