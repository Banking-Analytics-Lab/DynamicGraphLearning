import requests
import pandas as pd 

df = pd.read_csv('/home/emiliano/projects/def-cbravo/emiliano/RappiAnomalyDetection/dict.csv',header = None )
df.columns = ['State','Tag']
clean = lambda x: x.Tag.replace('.','').lower()
df.Tag = df.apply(clean,axis = 1)
count = 0 
for tag in df.Tag:
    try: 
        resp = requests.get(url = f'https://en.www.inegi.org.mx/contenidos/programas/ccpv/2020/tabulados/cpv2020_b_{tag}_01_poblacion.xlsx')


        output = open('test.xls', 'wb')
        output.write(resp.content)
        output.close()
        break
    except Exception as e: 
        print(f'{e=},\n{tag=}')
        count += 1 


print(count,len(df))


print(df.head())