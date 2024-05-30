import json

with open("dataset/test_output.json") as f:
    test_json = json.load(f)

datas = [entry["score"] for entry in test_json]
datas = sorted(datas)
shift_datas = datas[1:]
diff = [(shift_datas[i] - datas[i], i) for i in range(len(shift_datas))]
print(sorted(diff))
print(datas[310] - datas[290])
