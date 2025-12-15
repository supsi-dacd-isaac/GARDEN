from energyplus_runner import run_energyplus
from idf_generator import generate_idf
from parse_output_energyplus import parse_eso
import json


if __name__ == '__main__':

    input_directory = "input_files/automatically_generated"
    input_file = 'test_1.json'
    #read input file


    input_file = f"{input_directory}/{input_file}"
    with open(input_file, "r") as f:
        input = json.load(f)

    #generate idf file
    idf_file,input = generate_idf(input)

    #run energy+
    energyplus_output = run_energyplus(idf_file)

    #parse output from energy+
    df = parse_eso(energyplus_output, input)


    pass