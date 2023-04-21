# created by Nguyen Qui Vinh Quang.
#
# @Property of CVIPLab-2012
# @Contest: AIC2022
# Jisoo so cute

from allennlp.predictors.predictor import Predictor
import torch


class SRL(object):
    def __init__(self, weight_path):
        
        self.predictor = Predictor.from_path(weight_path)
        # print(weight_path)
        # exit(0)
        if torch.cuda.is_available():
            print("Using gpu")
            self.to_gpu()
        return

    def extract(self, query):
        return self.predictor.predict(sentence=query)

    def extract_json(self, query):
        return self.predictor.predict_batch_json(query)

    def to_gpu(self):
        self.predictor._model.to("cuda")


if __name__ == "__main__":
    # srl = SRL('src/pre_process/weight/structured-prediction-srl-bert.2020.12.15.tar.gz')
    # print(srl.extract('A black SUV runs down the street followed by another black vehicle.'))
    # nlp = dataset.NLP('pre_process/train-tracks.json')

    # data = nlp.get_data()
    # print(len(data))

    # outputs = []
    # begin_time = time.time()
    # for each_batch in data:
    #     outputs.extend(srl.extract_json(each_batch))
    # print("Total time:", time.time() - begin_time)
    # with open("pre_process/output.json", "w") as outfile:
    #     json.dump(outputs, outfile, indent=2)
    print("ok")
