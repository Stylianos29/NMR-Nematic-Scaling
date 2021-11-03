# This program creates as output for each input file the following:
# 1. A recovery pairs dictionary with the temperature values as keys
# pairing the headers of the recovery magnetization and recovery time 
# columns of the same temperature.
# 2. A .csv file simliar to the input file for which the values of the
# recovery times for each temperature are sorted in an ascenting
# fashion, with the coresponding same-temperature recovery magnetization
# values reorder appropriately.
# 3. A second .csv file, generated using the previous one, in which the
# recovery magnetization values at each temperature are normalized
# appropriately using the minimum and maximum values of each column.

import os
import time
import pandas as pd
import re
import pprint
import numpy as np
import shutil

def export_processed_dataframes_to_csv(beggining_of_filename, identifiers,
                                recovery_data_dictionary, output_directory):
    output_filename = (beggining_of_filename+identifiers+".csv")
    recovery_data_dataframe = pd.DataFrame( dict([ (k,pd.Series(v)) for k,v in
                                        recovery_data_dictionary.items() ]))
    recovery_data_dataframe.to_csv(os.path.join(output_directory,
                                output_filename), index = False, header=True)

# Input directory: All NMR raw data are stored in the 
# ".\NMR_Nematic_Scaling\NMR_measurements" directory, specified below
# in a platform-independent way
NMR_raw_data_relative_path = os.path.join("NMR_measurements")
# Full paths are specified user-independently as well
current_working_directory = os.getcwd()
NMR_raw_data_full_path = os.path.join(
    current_working_directory, NMR_raw_data_relative_path)

# Output directory: All the output files will be stored in the
# "./NMR_Nematic_Scaling/Output_Files/Processed_raw_data" directory
output_directory_relative_path = os.path.join("Output_Files",
                                                        "Processed_raw_data")
output_directory_full_path = os.path.join(
    current_working_directory, output_directory_relative_path)

# The last modification timestamps of the existing output directories
# need to be extracted for comparison purposes
output_data_last_modification_time_dictionary = {}
for output_directory in os.scandir(output_directory_full_path):
    if output_directory.is_dir():
        output_directory_identifying_parameters_tuple = tuple([
            float(number) for number in re.findall(r'-?\d+\.?\d*',
                str(output_directory))])
        output_data_last_modification_time_dictionary[
            output_directory_identifying_parameters_tuple] =(
                os.path.getmtime(output_directory))

new_directories_counter=0
for NMR_raw_data_file in os.scandir(NMR_raw_data_full_path):
    # PRELIMINARY CHECKS
    # All input files must be in .csv format
    if (NMR_raw_data_file.is_file() and (
            os.path.splitext(NMR_raw_data_file)[-1].lower() == '.csv')):

        # Comparison with timestamps of already existing directories in
        # the output directory
        input_file_last_modification_time = os.path.getmtime(NMR_raw_data_file)
        identifying_parameters_tuple = tuple(
            [float(number) for number in re.findall(r'-?\d+\.?\d*',
                                                    str(NMR_raw_data_file))])
        # If the last modification timestamp of the input file is
        # smaller than the correpsoning one of the output directory,
        # then skip processing this input file. Otherwise the output
        # files need to be updated. If the outpout files do not exist
        # yet though they need to be built properly.
        try:
            if (input_file_last_modification_time
                > output_data_last_modification_time_dictionary[
                    identifying_parameters_tuple]):
                print("Output for data set: x={} and H={} is updated.".format(
                        identifying_parameters_tuple[0],
                            identifying_parameters_tuple[1]) )
            else:
                continue
        except KeyError:
            print("A new output directory for data set:"
                    +" x={} and H={} is been created.".format(
                        identifying_parameters_tuple[0],
                            identifying_parameters_tuple[1]))

        # INPUT DATA PROCESSING
        # Passing the content of the .csv file to a Pandas DataFrame
        NMR_raw_magnetization_recovery_data_dataframe = pd.read_csv(
            NMR_raw_data_file.path)

        # Initialization of the recovery pairs dictionary
        NMR_raw_data_header_pairs_per_temperature_dictionary = {}

        # The header of each column of the raw data file contains
        # the temperature value for each measurement and needs to be
        # extrated initially in a list and then in a dictionary
        # sorted necessarily in ascending order. That will be the
        # recovery pairs dictionary. 
        NMR_dataframe_list_of_columns = list(
            NMR_raw_magnetization_recovery_data_dataframe.columns)
        list_of_temperatures_of_NMR_data = []
        for NMR_dataframe_column_header in NMR_dataframe_list_of_columns:
            pattern_containing_temperature_value = "Co5898_(.*?)K"
            temperature_value_string = re.search(
                pattern_containing_temperature_value,
                    NMR_dataframe_column_header).group(1)
            # In the original headers dots are marked with 'p'
            temperature_value_string = temperature_value_string.replace(
                                                                    'p','.')
            list_of_temperatures_of_NMR_data.append(float(
                                                    temperature_value_string))
        list_of_temperatures_of_NMR_data = sorted(set(
                                            list_of_temperatures_of_NMR_data))

        # Column headers for raw magnetization recovery and recovery
        # time for the same temperature value need to be paired and
        # assigned in an array of the form:
        # [raw magnetization recovery, recovery time] which will be the
        # value of the recovery pairs dictionary for each temperature
        # as the key value.
        for temperature in list_of_temperatures_of_NMR_data:
            # Converting back float temperature values to string with
            # dots turned to 'p' while removing any 'p's (dots) for the
            # case of integer temperature value
            temperature_string = '_'+str(temperature)+'K'
            temperature_string = temperature_string.replace('.','p')
            temperature_string = temperature_string.replace('p0','')
            # initializing the form of the array for the recovery pairs
            # dictionary values  
            headers_pair_per_temperature_array = ['Recovery time',
                'Recovery Magnetization']
            for column_header in NMR_dataframe_list_of_columns:
                if temperature_string in column_header:
                    # "twait_" at the beginning of the header
                    # indicates recovery time measurements
                    if 'twait_' in column_header:
                        headers_pair_per_temperature_array[0] = column_header
                    # "r_" at the beginning of the header
                    # indicates magnetization recovery measurements
                    else:
                        headers_pair_per_temperature_array[1] = column_header
            NMR_raw_data_header_pairs_per_temperature_dictionary[
                temperature] = headers_pair_per_temperature_array

        # Extracting the values of the pairs recovery magnetization-time
        # for each temperature
        sorted_NMR_magnetization_recovery_data_dictionary = {}
        normalized_NMR_magnetization_recovery_data_dictionary = {}   
        for temperature_value in (
                NMR_raw_data_header_pairs_per_temperature_dictionary.keys()):
            recovery_time = np.array(
                NMR_raw_magnetization_recovery_data_dataframe[
                    NMR_raw_data_header_pairs_per_temperature_dictionary[
                        temperature_value][0]])
            recovery_time = recovery_time[
                np.logical_not(np.isnan(recovery_time))]
            recovery_magnetization = np.array(
                NMR_raw_magnetization_recovery_data_dataframe[
                    NMR_raw_data_header_pairs_per_temperature_dictionary[
                        temperature_value][1]])
            recovery_magnetization = recovery_magnetization[
                np.logical_not(np.isnan(recovery_magnetization))]

            # Sorting recovery time values in ascenting order
            array_indices = recovery_time.argsort()
            recovery_time = recovery_time[array_indices[::1]]
            recovery_magnetization = recovery_magnetization[array_indices[::1]]

            # Passing the sorted values to the sorted data dictionary
            recovery_time_header = (
                NMR_raw_data_header_pairs_per_temperature_dictionary[
                    temperature_value][0])
            sorted_NMR_magnetization_recovery_data_dictionary[
                                        recovery_time_header] = recovery_time
            sorted_recovery_magnetization_header = (
                NMR_raw_data_header_pairs_per_temperature_dictionary[
                    temperature_value][1])
            sorted_NMR_magnetization_recovery_data_dictionary[
                sorted_recovery_magnetization_header] = recovery_magnetization

            # Normalizing the recovery magnetization data using formula
            # m(t) - min / (max - min)
            recovery_magnetization = (recovery_magnetization
                -min(recovery_magnetization)) /(max(recovery_magnetization)
                    -min(recovery_magnetization))

            # Passing the sorted values to the noralized data dictionary
            normalized_NMR_magnetization_recovery_data_dictionary[
                                        recovery_time_header] = recovery_time
            # normalized_recovery_magnetization_header = (
            #         "Normalized magnetization at T="+str(temperature_value))
            normalized_recovery_magnetization_header = (
                NMR_raw_data_header_pairs_per_temperature_dictionary[
                    temperature_value][1])
            normalized_NMR_magnetization_recovery_data_dictionary[
                normalized_recovery_magnetization_header
                ] = recovery_magnetization

        # OUTPUT DIRECTORY AND ITEMS
        # Exctracting the parameter values characteristic of the
        # raw data set that it's being processed
        raw_data_set_identifier = os.path.splitext(
                NMR_raw_data_file)[0].split(
                    "Raw Recovery Data ",1)[1].replace(' ','_H=')
        output_directory_of_input_file_full_path = os.path.join(
            output_directory_full_path,
                "Processed_raw_data_" + raw_data_set_identifier)
        
        if os.path.exists(output_directory_of_input_file_full_path):
            # This does what? #################################################<--------------
            shutil.rmtree(output_directory_of_input_file_full_path)
        os.makedirs(output_directory_of_input_file_full_path)
        new_directories_counter += 1
     
        # Exporting the contents of the recovery pairs dictionary to
        # a .txt file
        output_filename_of_recovery_pairs_per_temperature_dictionary = (
            "Recovery_magnetization-time_pairs_per_temperature_dictionary_"
                                            +raw_data_set_identifier+".txt")                                                        
        with open(os.path.join(output_directory_of_input_file_full_path,
                output_filename_of_recovery_pairs_per_temperature_dictionary)
                    ,'w') as data:
            data.write(
                str(NMR_raw_data_header_pairs_per_temperature_dictionary))

        # Exporting the content of the 'sorted' and 'normalized'
        # recovery data .csv files after they have been turned into
        # dataframes
        export_processed_dataframes_to_csv("Sorted_raw_data_",
            raw_data_set_identifier, 
                sorted_NMR_magnetization_recovery_data_dictionary,
                    output_directory_of_input_file_full_path)

        export_processed_dataframes_to_csv("Normalized_raw_data_",
            raw_data_set_identifier,
                normalized_NMR_magnetization_recovery_data_dictionary,
                    output_directory_of_input_file_full_path)

if new_directories_counter==0:
    print("No new output directories were created.")
else:
    print("A total of {} new directories were created or updated.".format(
                                                    new_directories_counter))