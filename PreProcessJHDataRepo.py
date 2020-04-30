import pandas as pd
import os
from myutils import user_yes_no_query

class PreProcessJHDataRepo:
    '''This is a class to read and process timeseries data from John Hopkins's Github Repository.'''

    # BASE_URL = 'https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/'
    # CONFIRMED = 'time_series_covid19_confirmed_global.csv'
    countries = {'AU':'Australia', 'USA':'US','UK':'United Kingdom', 'Spain':'Spain',
                 'S.Korea':'Korea, South', 'Italy':'Italy', 'Germany':'Germany',
                 'France':'France'};
    def __init__(self, jh_github_url='https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/', filename='time_series_covid19_confirmed_global.csv'):
        self.BASE_URL = jh_github_url
        self.CONFIRMED = filename
        self.path = 'Data/'

        url = self.BASE_URL + self.CONFIRMED
        self.df = pd.read_csv(url, error_bad_lines=False)

    def show_data_head(self):
        print(self.df.head())

    def check_data_directory(self):
        dir_name = self.path
        try:
            # Create target Directory
            os.mkdir(dir_name)
            print("Directory ", dir_name, " Created ")
        except FileExistsError:
            print("Directory ", dir_name, " already exists")
        # end try

    def process_data(self, dbg=False):
        self.check_data_directory()
        for key, country in self.countries.items():
            df = self.df.loc[self.df['Country/Region'] == country].copy()
            df = df.drop(columns=['Lat', 'Long'])

            total_rows = df.shape[0]  # gives number of row count
            if total_rows > 1:
                df.loc['sum'] = df.sum(axis=0)
                df.loc['sum', 'Country/Region'] = country
                df.loc['sum', 'Province/State'] = "Sum"
            #end for
            file = self.path + "TS-" + key + ".csv"
            df.to_csv(file, index=False)
            if dbg:
                print("CSV File Created for " + country + " at:\t" + file)
        #end for

        # Create CSV form Global Cases
        df = self.df.drop(columns=['Lat', 'Long'])
        df.loc['sum'] = df.sum(axis=0)
        file = self.path + "TS-Global.csv"
        df.to_csv(file, index=False)
        if dbg:
            print("CSV File Created for Global at:\t" + file)
        #end if
        print("\nDone: Preparing Data from Github Repository!")
    #end function

    def run(self, dbg=False):
        if dbg:
            self.show_data_head()
        #end if
        self.process_data(dbg=dbg)
   #end function run
#end class



if __name__ == '__main__':
    if user_yes_no_query("Do you really want to replace existing .csv files (where 0 new cases were replaced by avg.)?")==True:
        PreProcessJHDataRepo().run(True)
    else:
        print("Thank you for keeping existing data intact.")

