
import requests
import pandas as pd 

df = pd.read_csv('dict.csv',header = None )
df.columns = ['State','Tag']
clean = lambda x: x.Tag.replace('.','').lower()
df.Tag = df.apply(clean,axis = 1)
count = 0 
data = ['poblacion', 'fecundidad', 'mortalidad', 'migracion', 'caracteristicas_economicas']
nums = ['01', '02', '03','04','08']



for tag in df.Tag:
    print(tag)
    for i,dat in zip(nums,data): 
        
        try: 
            resp = requests.get(url = f'https://en.www.inegi.org.mx/contenidos/programas/ccpv/2020/tabulados/cpv2020_b_{tag}_{i}_{dat}.xlsx')
            print(i, dat)
            
            output = open(f'./scrapped/{tag}{i}{dat}.xlsx', 'wb')
            
            output.write(resp.content)
            output.close()
            
        except Exception as e: 
            print(f'{e=},\n{tag=}')
            count += 1 


print(count,len(df))


print(df.head())
