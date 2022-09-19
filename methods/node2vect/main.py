
# relative import flags
import os 
import sys
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.dirname(SCRIPT_DIR))
from Build_Data import dataBuilder
import argparse 
from pecanpy import pecanpy
import time


def main(data_output, data_input,buildData, run_Algo, alg_inp, alg_output): 
    if buildData: 
        builder = dataBuilder.Node2VectData(data_path= data_input)
        builder.build(output_file=data_output) 
    if run_Algo: 
        tic = time.perf_counter()
        g = pecanpy.SparseOTF(p=1, q=1, workers=10, verbose=True)
        g.read_edg(alg_inp, weighted=False, directed=False)
        g.simulate_walks(num_walks=1, walk_length=1)
        emb = g.embed()
        toc = time.perf_counter()
        print(f"Downloaded the tutorial in {toc - tic:0.4f} seconds")
        print(emb)



    

if __name__ == '__main__': 
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_output", help="output file for data builder",type=str, default = "")
    parser.add_argument("--data_input", help = 'input file to build data', type = str, default= "")
    parser.add_argument("--buildData", help ="Wheter to structure the data or just run Node2Vect", type = bool, default = False)
    parser.add_argument("--run_Algo" , help = "run algorithm or not" , type = bool,default = False)
    parser.add_argument("--alg_inp" , help = "node2vect inp, when running full pipeline should equal data_output", type = str, default ='')
    parser.add_argument("--alg_output" , help = 'path to algorithm output' , type = str, default  ='')
    arguments = vars(parser.parse_args())
    print(arguments)
    main(**arguments)

