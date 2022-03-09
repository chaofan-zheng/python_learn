import yaml
from pprint import pprint

yamlPath = 'test.yml'
f = open(yamlPath, 'r', encoding='utf-8')
content = yaml.load(f.read())
pprint(content)
f.close()

