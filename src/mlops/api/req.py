import requests
import json
response=requests.get(
    "https://api.github.com/repos/SkafteNicki/dtu_mlops"
    , params={'q': 'requests+language:python'}
    )

print(response.status_code)

if response.status_code == 200:
    print('Success!')
elif response.status_code == 404:
    print('Not Found.')
print(response.json())


response = requests.get('https://imgs.xkcd.com/comics/making_progress.png')
print(response.content)

pload = {'username':'Olivia','password':'123'}
response = requests.post('https://httpbin.org/post', data = pload)
print(response.content)