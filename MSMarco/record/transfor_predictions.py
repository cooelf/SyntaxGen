import json
# {"query_id": 10, "answers": ["No Answer Present."]}
f = open("predictions.json", 'r')
fw = open("prediction.json", 'w')
predict_dict = json.load(f)
for k, v in predict_dict.items():
    dict = {}
    dict["query_id"] = int(k)
    dict["answers"] = [v]
    data = json.dumps(dict)
    fw.writelines(data)
    fw.writelines("\n")

f.close()
fw.close()