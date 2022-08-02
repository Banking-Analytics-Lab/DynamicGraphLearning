from operator import index
import pandas as pd 


df = pd.read_csv('dict.csv',header = None )
df.columns = ['State','Tag']
clean = lambda x: x.Tag.replace('.','').lower()
df.Tag = df.apply(clean,axis = 1)


population_dict = {}
population_male_female_dict = {}
cities_dict = {}
fertility_dict = {}
mortality_dict = {}
migration_dict = {}
economics_dict = {}

for tag in df.Tag:
    population = pd.read_excel(f'./scrapped/{tag}01poblacion.xlsx', sheet_name = '01', skiprows = 8, header = None) 
    population.columns = ["Entidad federativa", "Municipio", "Localidades/Población", "Total de localidades y población1", "Tamaño de localidad: 1-249 habitantes", 	
    "Tamaño de localidad: 250-499 habitantes",		"Tamaño de localidad: 500-999 habitantes","Tamaño de localidad: 1 000-2 499 habitantes","Tamaño de localidad: 2 500-4 999 habitantes",
    "Tamaño de localidad: 5 000-9 999 habitantes","Tamaño de localidad: 10 000-14 999 habitantes","Tamaño de localidad: 15 000-29 999 habitantes",
    "Tamaño de localidad: 30 000-49 999 habitantes","Tamaño de localidad: 50 000-99 999 habitantes","Tamaño de localidad: 100 000-249 999 habitantes",
    "Tamaño de localidad: 250 000-499 999 habitantes","Tamaño de localidad: 500 000-999 999 habitantes","Tamaño de localidad: 1 000 000 y más habitantes"]

    population_dict[population['Entidad federativa'][1]] = population['Total de localidades y población1'][1]
    population_df = pd.DataFrame.from_dict(population_dict, orient='index').reset_index()
    population_df.columns = ['State','Population']

    cities_dict[population['Entidad federativa'][0]] = population['Total de localidades y población1'][0]
    cities_df = pd.DataFrame.from_dict(cities_dict, orient='index').reset_index()
    cities_df.columns = ['State','# cities']

    population_male_female = pd.read_excel(f'./scrapped/{tag}01poblacion.xlsx', sheet_name = '03', skiprows = 8, header = None) 
    population_male_female.columns = ['Entidad federativa','Municipio','Edad desplegada','Población total1','Sexo: Hombres','Sexo: Mujeres','Relación hombres-mujeres2']
    population_male_female_dict[population_male_female['Entidad federativa'][0]] = population_male_female['Relación hombres-mujeres2'][0]
    population_male_female_df = pd.DataFrame.from_dict(population_male_female_dict, orient='index').reset_index()
    population_male_female_df.columns = ['State','Male-female ratio']


    fertility = pd.read_excel(f'./scrapped/{tag}02fecundidad.xlsx', sheet_name = '01', skiprows=8, header = None)
    fertility.columns = ['Entidad federativa','Tamaño de localidad','Grupos quinquenales de edad','Población femenina de 12 años y más','Número de hijas e hijos nacidos vivos: 0',
    'Número de hijas e hijos nacidos vivos: 1',
    'Número de hijas e hijos nacidos vivos: 2', 'Número de hijas e hijos nacidos vivos: 3','Número de hijas e hijos nacidos vivos: 4','Número de hijas e hijos nacidos vivos: 5',
    'Número de hijas e hijos nacidos vivos: 6','Número de hijas e hijos nacidos vivos: 7','Número de hijas e hijos nacidos vivos: 8','Número de hijas e hijos nacidos vivos: 9',
    'Número de hijas e hijos nacidos vivos: 10','Número de hijas e hijos nacidos vivos: 11','Número de hijas e hijos nacidos vivos: 12','Número de hijas e hijos nacidos vivos: 13 y más',
    'Número de hijas e hijos nacidos vivos: No especificado1', 'Hijas e hijos nacidos vivos2: Total','Hijas e hijos nacidos vivos2: Promedio3']

    fertility_dict[fertility['Entidad federativa'][0]] = fertility['Hijas e hijos nacidos vivos2: Promedio3'][0]
    fertility_df = pd.DataFrame.from_dict(fertility_dict, orient='index').reset_index()
    fertility_df.columns = ['State', '# kids per person']

    mortality = pd.read_excel(f'./scrapped/{tag}03mortalidad.xlsx', sheet_name='01', skiprows=8, header = None )
    mortality.columns = ['Entidad federativa','Tamaño de localidad','Grupos quinquenales de edad','Total de hijas e hijos nacidos vivos de la población femenina de 12 años y más1',
    'Hijas e hijos fallecidos: Total','Hijas e hijos fallecidos: Porcentaje']
    mortality_dict[mortality['Entidad federativa'][0]] = mortality['Hijas e hijos fallecidos: Porcentaje'][0]
    mortality_df = pd.DataFrame.from_dict(mortality_dict, orient='index').reset_index()
    mortality_df.columns = ['State', 'percent infant mortality']

    migration = pd.read_excel(f'./scrapped/{tag}04migracion.xlsx', sheet_name = '06', skiprows = 8, header = None )
    migration.columns = ['Entidad federativa','Tamaño de localidad','Sexo','Grupos quinquenales de edad','Población de 5 años y más migrante1',
    'Causa de la migración entre marzo de 2015 y marzo de 2020: Buscar trabajo','Causa de la migración entre marzo de 2015 y marzo de 2020: Cambio u oferta de trabajo',
    'Causa de la migración entre marzo de 2015 y marzo de 2020: Reunirse con la familia','Causa de la migración entre marzo de 2015 y marzo de 2020: Se casó o unió',
    'Causa de la migración entre marzo de 2015 y marzo de 2020: Estudiar','Causa de la migración entre marzo de 2015 y marzo de 2020: Por inseguridad delictiva o violencia',
    'Causa de la migración entre marzo de 2015 y marzo de 2020: Por desastres naturales','Causa de la migración entre marzo de 2015 y marzo de 2020: Lo deportaron',
    'Causa de la migración entre marzo de 2015 y marzo de 2020: Otra causa','Causa de la migración entre marzo de 2015 y marzo de 2020: No especificado']
    migration_dict[migration['Entidad federativa'][0]] = migration['Población de 5 años y más migrante1'][0]
    migration_df = pd.DataFrame.from_dict(migration_dict, orient='index').reset_index()
    migration_df.columns = ['State', 'Migrant population aged 5 and over']

    economics = pd.read_excel(f'./scrapped/{tag}08caracteristicas_economicas.xlsx', sheet_name = '01', skiprows = 9, header = None)
    economics.columns = ['Entidad federativa','Tamaño de localidad','Sexo','Grupos quinquenales de edad','Población de 12 años y más',
    'Condición de actividad económica: Población económicamente activa: Total',
    'Condición de actividad económica: Población económicamente activa: Ocupada','Condición de actividad económica: Población económicamente activa: Desocupada',
    'Condición de actividad económica: Población no económicamente activa',
    'Condición de actividad económica: No especificado','Tasa específica de participación económica1']
    economics_dict[economics['Entidad federativa'][0]] = economics['Tasa específica de participación económica1'][0]
    economics_df = pd.DataFrame.from_dict(economics_dict, orient='index').reset_index()
    economics_df.columns = ['State', 'Specific rate of economic participation']

   

final = pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(pd.merge(population_df,population_male_female_df, how = 'inner'),fertility_df, how = 'inner'),mortality_df, how = 'inner'), migration_df, how = 'inner'),
economics_df, how = 'inner'),  cities_df, how = 'inner')
final.to_excel('./scrapped/Final data for clustering.xlsx', index = False)






