import os

# This script builds the sample webpage.

if __name__ == '__main__':
    result = "<html><head><title>These words were never spoken.</title></head><body><h1>Handpicked results</h1>"
    for fv in os.listdir('../../results/favorites'):
        url = f'https://github.com/neonbjb/tortoise-tts/raw/main/results/favorites/{fv}'
        result = result + f'<audio controls="" style="width: 600px;"><source src="{url}" type="audio/mp3"></audio><br>\n'

    result = result + "<h1>Handpicked longform result:<h1>"
    url = f'https://github.com/neonbjb/tortoise-tts/raw/main/results/favorite_riding_hood.mp3'
    result = result + f'<audio controls="" style="width: 600px;"><source src="{url}" type="audio/mp3"></audio><br>\n'

    result = result + "<h1>Compared to Tacotron2 (with the LJSpeech voice):</h1><table><th>Tacotron2+Waveglow</th><th>TorToiSe</th>"
    for k in range(2,5,1):
        url1 = f'https://github.com/neonbjb/tortoise-tts/raw/main/results/tacotron_comparison/{k}-tacotron2.mp3'
        url2 = f'https://github.com/neonbjb/tortoise-tts/raw/main/results/tacotron_comparison/{k}-tortoise.mp3'
        result = result + f'<tr><td><audio controls="" style="width: 300px;"><source src="{url1}" type="audio/mp3"></audio><br>\n</td>' \
                          f'<td><audio controls="" style="width: 300px;"><source src="{url2}" type="audio/mp3"></audio><br>\n</td></tr>'
    result = result + "</table>"

    result = result + "<h1>Various spoken texts for all voices:<h1>"
    voices = ['angie', 'daniel', 'deniro', 'emma', 'freeman', 'geralt', 'halle', 'jlaw', 'lj', 'myself',
              'pat', 'snakes', 'tom', 'train_atkins', 'train_dotrice', 'train_kennard', 'weaver', 'william']
    lines = ['<table><th>text</th>' + ''.join([f'<th>{v}</th>' for v in voices])]
    line = f'<tr><td>reference clip</td>'
    for v in voices:
        url = f'https://github.com/neonbjb/tortoise-tts/raw/main/voices/{v}/1.wav'
        line = line + f'<td><audio controls="" style="width: 150px;"><source src="{url}" type="audio/mp3"></audio></td>'
    line = line + "</tr>"
    lines.append(line)
    for txt in os.listdir('../../results/various/'):
        if 'desktop' in txt:
            continue
        line = f'<tr><td>{txt}</td>'
        for v in voices:
            url = f'https://github.com/neonbjb/tortoise-tts/raw/main/results/various/{txt}/{v}.mp3'
            line = line + f'<td><audio controls="" style="width: 150px;"><source src="{url}" type="audio/mp3"></audio></td>'
        line = line + "</tr>"
        lines.append(line)
    result = result + '\n'.join(lines) + "</table>"

    result = result + "<h1>Longform result for all voices:</h1>"
    for lf in os.listdir('../../results/riding_hood'):
        url = f'https://github.com/neonbjb/tortoise-tts/raw/main/results/riding_hood/{lf}'
        result = result + f'<audio controls="" style="width: 600px;"><source src="{url}" type="audio/mp3"></audio><br>\n'

    result = result + "</body></html>"
    with open('result.html', 'w', encoding='utf-8') as f:
        f.write(result)
