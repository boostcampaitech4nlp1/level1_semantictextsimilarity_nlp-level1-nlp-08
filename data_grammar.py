import requests
import json
import pandas as pd
from tqdm import tqdm

if __name__ == '__main__' :
    df = pd.read_csv('./data/train.csv')

    column = ['sentence_1', 'sentence_2']
    row_count = len(df)

    for series in column :
        replace_series = pd.Series(dtype=str)
        for i in tqdm(range(0, row_count, 20) , desc=series, mininterval=0.01) :
            sentence_series = df[series][i:i+20]
            text = ''
            for item in sentence_series :
                text = text+ item + '\r\n'
            
            load_error = False
            try :
                response = requests.post('http://164.125.7.61/speller/results', data={'text1': text})
                data = response.text.split('data = [', 1)[-1].rsplit('];', 1)[0]
                data = json.loads(data)
            except :
                load_error = True
                print(text)

            if load_error == False :
                for idx,err in enumerate(data['errInfo']) :
                    err['candWord'] = err['candWord'].split('|')[0]
                    text= text.replace(err['orgStr'] , err['candWord'])
                
        
            temp = pd.Series(text.split('\r\n')[:-1])
            replace_series =pd.concat([replace_series,temp])

        replace_series = replace_series.reset_index(drop=True)
        df[series] = replace_series
        
            
    df.to_csv('replace_train.csv',index=False)
    

    
    
