from pysolr import Solr
import json

solr_url = 'http://localhost:8983/solr/ir_final'
solr = Solr(solr_url)

# Load JSON data from file
with open("integrated_data_latest.json", 'r') as file:
    json_data_list = json.load(file)

print("loading the data is done")
# Set to track uploaded documents
# s = set()
count = 0
for json_data in json_data_list:
    try:
        del json_data["_version_"]
        if "anchor" in json_data:
            del json_data["anchor"]
        if "cache" in json_data:
            del json_data["cache"]
        solr.add(json_data)
        count += 1
        if count % 10000 == 0:
            print(f"{count} are uploaded successfully")
    except Exception as e:
        print(f'{e} has occurred with id: {json_data["id"]}')



# Commit remaining changes
solr.commit()
# print(count)
