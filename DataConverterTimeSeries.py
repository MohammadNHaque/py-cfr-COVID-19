import pandas as pd
import numpy as np
import os
from os import path
import sys
from myutils import user_yes_no_query, check_directory

class DataConverterTimeSeries:
    '''This is a class to  convert the time series to dataset format required for the cfrV2.'''

    def __init__(self, lst_countries=[]):
        self.path = 'Data/Dataset/'
        self.ts_path = 'Data/'
        if len(lst_countries) == 0:
            self.countries = ["AU", "USA", "UK", "Spain", "S.Korea", "Italy", "Germany", "France", "Global"]
        else:
            self.countries = lst_countries

    def check_country_files(self, dbg=False):
        for country in self.countries:
            file = self.ts_path + "TS-" + country + ".csv"
            if not path.isfile(file):
                print("File not Found: " + file)
                print("Please Run PreProcessJHDataRepo.py to create the TimeSeries Data.")
                sys.exit(-1)
            elif dbg: print("File Found at: " + file)
            #end if
        #end for

    def run(self, dbg=False):
        check_directory(self.path, dbg=dbg)
        self.check_country_files(dbg=dbg)

        if dbg:
            print("\nStart: Creating Dataset ")
        for country in self.countries:
            # read data for the country
            file = self.ts_path + "TS-" + country + ".csv"
            data = pd.read_csv(file)
            data.drop(['Province/State', 'Country/Region'], axis=1, inplace=True)

            tot_rows = data.shape[0]
            if tot_rows > 1:
                del_rows = range(data.shape[0] - 1)
                l = [*del_rows]
                data = data.drop(l, axis=0)
            # end if

            # transpose the TS data
            df_transposed = data.T
            df_transposed.reset_index(inplace=True)
            df_transposed.columns = ["Date", "Confirmed"]
            df = df_transposed

            # Convert Time Series to Dataset format
            row_max = df.shape[0] - 14
            index = []
            dataf = np.zeros((row_max, 15), dtype='int')

            for row in range(0, row_max):
                dataf[row][0] = df.iloc[row + 14][1]
                index.append(df.iloc[row + 14][0])
                for col in range(14, 0, -1):
                    dataf[row][col] = df.iloc[row + (14 - col)][1]
                # end for x
            # end for y

            # Convert numpy to pandas dataframe
            pdf = pd.DataFrame(data=dataf, index=index,
                               columns=["Confirmed", "D1", "D2", "D3", "D4", "D5", "D6", "D7", "D8", "D9", "D10", "D11",
                                        "D12", "D13", "D14"])

            # write pd DataFrame to csv
            pdf.to_csv(self.path + "DS-" + country + ".csv", index=True)

            # locate first non-zero row
            nz_locs = pdf['D14'] != 0
            df_nz = pdf.loc[nz_locs, :]
            # save two format: one with Date as index and another without index
            df_nz.to_csv(self.path + "DS-cfr-" + country + ".csv", index=False) #--- to use with cfr program
            df_nz.to_csv(self.path + "DS-non0-" + country + ".csv", index=True)
            if dbg:
                print("Done: Dataset Creation for " + country)
            #end if

        # end for countries
        print("\nDone: Converting Time Series to Dataset!")
    #end function run
#end class

if __name__ == '__main__':
    if user_yes_no_query("Do you really want to replace existing .csv files (where 0 new cases were replaced by avg.)?")==True:
        DataConverterTimeSeries().run(True)
    else:
        print("Thank you for keeping existing data intact.")

